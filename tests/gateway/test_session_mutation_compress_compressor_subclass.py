"""A plugin engine that subclasses ContextCompressor still owns gateway ``/compress``.

Regression for #147875: ``isinstance(..., ContextCompressor)`` treated every subclass as the
built-in compressor. ``_plugin_engine`` then dropped a resident plugin, and a cold one-off
subclass never received ``on_session_end``.
"""
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor
from gateway.session_mutation_compress import _plugin_engine


class _PluginCompressor(ContextCompressor):
    """Third-party engine that extends the built-in compressor."""

    def __init__(self, *, noop=False):
        self.noop = noop
        self.events = []

    def update_model(self, **kwargs):
        self.events.append(('update_model', kwargs.get('model')))

    def bind_session_state(self, session_db=None, session_id=''):
        self.events.append(('bind', session_id))

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=''):
        self.events.append(('compress', focus_topic, force, len(messages)))
        if self.noop:
            return messages
        return [{'role': 'user', 'content': 'PLUGIN_SUMMARY', '_compressed_summary': True}]

    def on_session_start(self, session_id, **kwargs):
        self.events.append(('start', session_id, kwargs.get('boundary_reason')))

    def on_session_end(self, session_id, messages):
        self.events.append(('end', session_id, len(messages)))


def test_plugin_engine_returns_a_compressor_subclass_and_not_the_builtin():
    plugin = _PluginCompressor()
    builtin = ContextCompressor.__new__(ContextCompressor)
    kwargs = dict(authority=None, policy=None, config={}, model='m', runtime={}, target='sid')

    assert _plugin_engine(resident=SimpleNamespace(context_compressor=plugin), **kwargs) is plugin
    assert _plugin_engine(resident=SimpleNamespace(context_compressor=builtin), **kwargs) is None
    assert _plugin_engine(resident=SimpleNamespace(context_compressor=None), **kwargs) is None
    assert _plugin_engine(resident=SimpleNamespace(), **kwargs) is None


def _compress_session(tmp_path, monkeypatch, *, resident, turns):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    import hermes_state_runtime as rt

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    runner = SimpleNamespace(
        session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: resident,
        _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(
        request_id='subclass', source='cli', cwd=str(tmp_path), model='frozen', toolsets=[]))
    for i in range(turns):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}')
    return owner, db, ref, db.get_session(ref.session_id)


async def _compress(owner, ref, before):
    reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': ref.session_id, 'request_id': 'compress',
        'expected_revision': before['runtime_revision'],
        'expected_generation': before['runtime_generation'],
        'operation': 'compress', 'payload': {'focus': 'the plan'}}})
    assert reply.get('error') is None, reply
    return reply.get('result') or {}


def _named(events, kind):
    return [event for event in events if event[0] == kind]


@pytest.mark.asyncio
async def test_resident_subclass_runs_on_compress(tmp_path, monkeypatch):
    constructed = []
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: constructed.append(self))
    monkeypatch.setattr(ContextCompressor, 'compress', lambda self, messages, **kw: [
        {'role': 'user', 'content': 'BUILTIN_SUMMARY', '_compressed_summary': True}])
    plugin = _PluginCompressor()
    owner, db, ref, before = _compress_session(
        tmp_path, monkeypatch, resident=SimpleNamespace(context_compressor=plugin), turns=4)
    try:
        result = await _compress(owner, ref, before)
    finally:
        await owner.close()
    target = result.get('target_session_id', ref.session_id)
    assert _named(plugin.events, 'compress') == [('compress', 'the plan', True, 8)]
    assert constructed == []
    assert 'PLUGIN_SUMMARY' in str(db.get_messages_as_conversation(target))
    # The resident agent owns the engine: /compress announces the boundary and does not end it.
    assert _named(plugin.events, 'start') == [('start', target, 'compression')]
    assert _named(plugin.events, 'end') == []


@pytest.mark.asyncio
async def test_resident_builtin_compressor_is_replaced_for_compress(tmp_path, monkeypatch):
    constructed = []
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: constructed.append(self))
    monkeypatch.setattr(ContextCompressor, 'compress', lambda self, messages, **kw: [
        {'role': 'user', 'content': 'BUILTIN_SUMMARY', '_compressed_summary': True}])
    builtin_calls = []
    builtin = ContextCompressor()
    builtin.compress = lambda messages, **kw: builtin_calls.append(len(messages)) or messages
    fresh = len(constructed)
    owner, db, ref, before = _compress_session(
        tmp_path, monkeypatch, resident=SimpleNamespace(context_compressor=builtin), turns=4)
    try:
        result = await _compress(owner, ref, before)
    finally:
        await owner.close()
    target = result.get('target_session_id', ref.session_id)
    assert builtin_calls == []
    assert len(constructed) == fresh + 1
    text = str(db.get_messages_as_conversation(target))
    assert 'BUILTIN_SUMMARY' in text
    assert 'PLUGIN_SUMMARY' not in text


@pytest.mark.asyncio
async def test_cold_builtin_noop_does_not_end_the_compressor(tmp_path, monkeypatch):
    ended = []
    monkeypatch.setattr('agent.agent_init._select_context_engine', lambda config: None)
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', lambda self, messages, **kw: messages)
    monkeypatch.setattr(
        ContextCompressor, 'on_session_end',
        lambda self, session_id, messages: ended.append(session_id))
    owner, _db, ref, before = _compress_session(tmp_path, monkeypatch, resident=None, turns=1)
    try:
        result = await _compress(owner, ref, before)
    finally:
        await owner.close()
    assert result.get('status') == 'noop', result
    assert ended == []


@pytest.mark.asyncio
async def test_cold_subclass_noop_ends_the_one_off_engine(tmp_path, monkeypatch):
    monkeypatch.setattr('agent.model_metadata.get_model_context_length', lambda *a, **k: 128_000)
    plugin = _PluginCompressor(noop=True)
    monkeypatch.setattr('agent.agent_init._select_context_engine', lambda config: plugin)
    owner, _db, ref, before = _compress_session(tmp_path, monkeypatch, resident=None, turns=1)
    try:
        result = await _compress(owner, ref, before)
    finally:
        await owner.close()
    assert result.get('status') == 'noop', result
    assert _named(plugin.events, 'compress') == [('compress', 'the plan', True, 2)]
    assert _named(plugin.events, 'end') == [('end', ref.session_id, 2)]
    assert _named(plugin.events, 'start') == []


@pytest.mark.asyncio
async def test_cold_subclass_compress_notifies_and_ends_the_engine(tmp_path, monkeypatch):
    monkeypatch.setattr('agent.model_metadata.get_model_context_length', lambda *a, **k: 128_000)
    plugin = _PluginCompressor()
    monkeypatch.setattr('agent.agent_init._select_context_engine', lambda config: plugin)
    owner, db, ref, before = _compress_session(tmp_path, monkeypatch, resident=None, turns=4)
    try:
        result = await _compress(owner, ref, before)
    finally:
        await owner.close()
    target = result.get('target_session_id', ref.session_id)
    assert result.get('status') != 'noop', result
    assert _named(plugin.events, 'compress') == [('compress', 'the plan', True, 8)]
    assert _named(plugin.events, 'start') == [('start', target, 'compression')]
    assert _named(plugin.events, 'end') == [('end', target, 1)]
    assert 'PLUGIN_SUMMARY' in str(db.get_messages_as_conversation(target))
