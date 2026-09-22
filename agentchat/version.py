"""Bridge protocol version — leaf module so any part of the package can
import it without cycles.

Reported to the backend at executor registration and WS gateway join
(audit-remediation-plan H3). The backend compares it against
Agentchat.Protocol's @min_bridge_version and refuses/flags outdated bridges
once enforcement is on. Bump on any protocol-relevant change (payload fields
the backend requires, structured-tag semantics, fail-loud contracts).
Distinct from the ACP message-envelope schema_version ("2.0") — they version
different things.
"""

# 2.2.0 — task-request sequencing moved server-side (H4 item 4, issue #86):
# the bridge submits parsed <task_request> blocks to
# POST /api/gateway/task-requests instead of running the orchestrator
# scope→create flow and default-assignee policy locally.
# 2.3.0 — compound-task DAG walk moved server-side (H4 item 4 flows 2–3):
# the bridge loops on POST /api/gateway/tasks/:id/claim-step and only
# runs the per-step LLM; the dead memory_flush handler (no producer,
# zero tasks ever created) was deleted outright.
# 2.4.0 — prompt-cache restructure: per-turn-fresh blocks (temporal
# context, live presence, speaking order) moved out of promptDirectives
# into directives.volatileContext, which the bridge appends to the USER
# turn. A 2.3.x bridge on a 2.4 backend would silently lose those blocks.
# 2.4.1 — cross-turn history cache continuity: cache breakpoint pinned at
# the stable-history boundary (per-turn tail — volatile context, trigger
# echo, identity anchor — moved after it), anchored non-sliding history
# window in _cached_get_messages, and trigger echo deduped when it already
# rendered as the newest history message. Bridge-internal (no backend
# payload change), but listed for fleet-roll tracking.
# 2.5.0 — per-turn model override consumed: the bridge now applies
# task metadata `model_override` (stamped by PulseExecutionWorker from
# pulse config `model`) to every LLM call in that task's turn via the
# MODEL_OVERRIDE contextvar; backends resolve it at request time
# (_request_model). Older bridges silently ran pulses on the agent's
# static model.
# 2.6.0 — humanlike bubble delivery moved fully server-side (audit
# Theme 5.3): the bridge posts its raw <msg>-tagged reply ONCE; the
# backend (HumanlikeDelivery + StaggeredBubbleWorker) owns splitting,
# pacing, humanlike_bubble metadata, and peer-wake routing. Older
# bridges split client-side and make their own peer-wake routing
# decision, so WS/SDK agents and bridge agents diverge; the
# behavioralConfig.humanlikePacing key they read is gone (they fall
# back to local defaults, harmless during the roll).
# 2.6.1 — open-subtask completion rejection treated as a wait signal:
# when the backend's complete_task guard answers with the
# "[open_subtasks]" marker, _handle_task leaves the task open for the
# sub-task-completion wake instead of failing the task. Older bridges
# fail the root task and strand the sub-tasks' output (Morning Brief
# 2026-08-12). Backend marker ships in the same change; safe either
# order — without the marker the old failure mode simply persists.
# 2.7.0 — runnability reporting: the bridge preflights its model backend
# before registering and sends `backend_health` {status, detail} on both
# the register and heartbeat payloads, plus `bridge_version` on heartbeat.
# The server (Agentchat.Agents.Runnability) gates agent presence on it, so
# a machine with no `claude login` now reads offline-with-a-reason and gets
# an in-chat explanation instead of a green dot and permanent silence.
# claude_cli additionally classifies auth-shaped CLI failures as
# BackendAuthError and flips its own health, so a mid-session credential
# loss (or recovery) surfaces within one heartbeat. Older bridges simply
# don't report health — they're treated as healthy on the version check
# alone, so the roll is safe in either order.
# 2.7.1 — preflight probes the Bedrock/Vertex credential chain instead of
# assuming it's configured. The exemption meant a cloud-connection agent on
# a machine with no AWS/GCP chain reported `ok` forever: every turn died on
# "Could not load credentials from any providers", and every restart re-ran
# preflight and laundered the dead state back to green, so it kept reading
# online and kept getting handed tasks. Structural probe only (no network,
# so an instance-role-only machine now reads unauthenticated — the safe
# direction). Not gated on by the server; the roll is safe in either order.
# 2.7.2 — audit hardening: missing agntchat_mcp_server.py is a hard error
# for claude_cli tool use (no silent XML-loop degrade), the stale repo
# scripts/ fallback paths were removed from every script lookup, WS event
# handlers hold strong task refs (GC could silently drop events parked on
# the semaphore), the dead batch_complete_tasks stub was deleted, and
# heartbeat/location failure paths now log. Bridge-internal; not gated on
# by the server, safe to roll in either order.
# 2.7.4 — subscription preflight no longer counts a bare ~/.claude.json as
# a credential: it's the CLI's config file, written on any first launch,
# so a fresh machine that never ran `claude login` read `ok` forever and
# the first-run onboarding "greeting" step stalled with no warning. The
# probe now looks for account markers (oauthAccount / primaryApiKey keys
# only, never values) and recognises CLAUDE_CODE_OAUTH_TOKEN. Not gated on
# by the server; safe to roll in either order.
# 2.7.5 — turn-time auth failures reach the user: _AUTH_FAILURE_RE learns
# the CLI's "Failed to authenticate" / "OAuth session expired" phrasings
# (previously only "authentication failed" / "oauth token expired", so an
# expired login posted the generic modelFailure apology and never flipped
# health), and BackendAuthError turns now reply with the server's
# errorMessages.authFailure copy instead of the generic fallback. Pairs
# with the backend adding that key; older backends fall back to a built-in
# string, safe to roll in either order.
# 2.7.6 — authFailure replies carry metadata.errorKind="auth_failure" so
# clients can render a one-click fix (the desktop's "Sign in to Claude"
# button) under the error bubble instead of leaving the user to parse the
# copy. Pure metadata addition; clients that don't know the key ignore it
# and the server stores it as-is, safe to roll in either order.
# 2.7.7 — bedrock/vertex auth failures are no longer silent. Two bugs, one
# incident (Jarvis/Bedrock, Aug 2026): an expired AWS SSO session's error
# text ("Token is expired. To refresh this SSO session run 'aws sso login'
# ...") didn't match _AUTH_FAILURE_RE at all, so it fell through to a plain
# RuntimeError — health never flipped, and the turn eventually posted the
# generic "I ran into an issue" apology ~3 minutes later (the CLI's own AWS
# auth-refresh timeout) with zero indication of what actually broke. Even
# once classified correctly, the existing authFailure copy ("sign in to
# Claude") is wrong for a cloud-authenticated agent. Fixed: the regex learns
# AWS SSO / GCP ADC phrasings, and _model_failure_reply now branches on the
# backend's cli_connection — subscription keeps the existing copy+button,
# bedrock/vertex get the new errorMessages.authFailureCloud copy with the
# CLI's own (already-correct) remedy text appended verbatim. Pairs with the
# backend adding that key; older backends fall back to a built-in string,
# safe to roll in either order.
# 2.8.0 — Claude usage/rate-limit turns are classified distinctly from
# credential failures and generic errors. New BackendRateLimitError
# (claude_cli.py: _RATE_LIMIT_RE + best-effort _extract_reset_time;
# anthropic.py: RateLimitError / APIStatusError 429/529) reports
# backend_status "rate_limited" — self-healing, same as BackendAuthError's
# "unauthenticated" clears on the next successful turn — and replies with
# the server's errorMessages.rateLimitFailure copy, appending a "resumes
# around HH:MM" ETA when one could be determined. Turns also carry
# metadata.errorKind="rate_limit" (no button, unlike auth_failure — there's
# nothing to click to fix a usage limit). Before this, a usage limit fell
# into the generic modelFailure apology, indistinguishable from a real bug,
# and invited retrying during the exact window that burns more of the same
# quota. Pairs with the backend's Runnability :llm_rate_limited blocker
# code and errorMessages.rateLimitFailure key; older backends fall back to
# a built-in string, safe to roll in either order.
# 2.9.0 — MCP routing context moved into the MCP_CONTEXT contextvar
# (backends/__init__.py), off the shared backend instance's mutable
# _mcp_* attributes. Prerequisite for max_concurrent > 1: with two
# in-flight turns, instance attributes interleave write→write→read→read
# and turn A's tool calls route into turn B's conversation/task. NOT
# safe to roll in either order — a pre-2.9 bridge handed 2 slots has the
# race live, so the backend raises @min_bridge_version to 2.9.0 in the
# same change that flips the max_concurrent_tasks default to 2.
# 2.9.1 — every REST call carries X-Bridge-Version (rest.py _request).
# The backend's /api/agents/my/settings clamps max_concurrent_tasks to 1
# for callers that don't prove the 2.9.0 floor; since pre-2.9.1 bridges
# never send the header, the concurrency floor now holds by construction
# instead of resting on the enforce_bridge_version feature flag staying
# on. Safe to roll in either order: an older backend ignores the header,
# a newer backend clamps older bridges to the single slot they had
# before 2.9.0 anyway.
# 2.9.2 — rate_limited health carries a reset deadline. backend_health
# gains `reset_at` (unix epoch): the parsed ETA when the CLI/API error
# named one, else now + RATE_LIMIT_RESET_FALLBACK_SECONDS. Closes the
# stale-offline gap in the 2.8.0 self-healing story: the blocker only
# cleared on a *successful turn*, so an agent that got no traffic after
# the limit reset read offline indefinitely. The server (Runnability)
# now expires an :llm_rate_limited blocker once reset_at passes; if the
# guess was early the next turn fails fast and re-marks with a fresh
# deadline. Safe to roll in either order: an older backend ignores the
# extra key, and an older bridge sends no reset_at, which keeps the
# pre-2.9.2 clear-on-success behaviour for that executor.
#
# 2.9.3 — a silent `end_turn` (reason no_action_needed / thread_redirect)
# drops any prose the model emitted alongside it instead of posting the
# declined turn ("…nothing for me to add here", conv 0b86e6ed). The claude_cli
# stream parser now keeps each tool use's parsed input so the reason is
# readable after the run. Terminator reasons keep their text.
# 2.9.4 — coordination-audit wave 2 follow-ups (2026-09-06 audit, "Bridge
# follow-ups"): every `/api/mcp` tools/call carries the caller context as
# `params._meta.context` (complete_task/fail_task send task_id + the acting
# conversation; search_memory its conversation), the hidden thread redirect
# posts the canonical EndTurn JSON `{"reason": "thread_redirect", "message"}`
# instead of prose, a routed `<dm>` keeps the remaining group text (hidden
# redirect / failure notice only fill an otherwise-empty turn — identical to
# the server-side router), the message dedup TTL covers the backend's
# 1800 s claim hard cap (MESSAGE_DEDUP_TTL_SECONDS), and
# update_task_status(complete|failed) raises before the network instead of
# collecting the backend's 422. Not gated on by the server; the backend
# tolerates the older wire shapes, safe to roll in either order.
# 2.9.5 — coordination-audit wave 4 (07-tool-surface "permission_prompt
# blocks the whole CLI turn … and denies on timeout"): a permission request
# that EXPIRES (backend status `expired`, or the bridge's local poll ceiling)
# now reaches the model as "expired without an answer — ask again later or
# proceed without it" (PERMISSION_EXPIRED_MESSAGE) instead of a denial; the
# wire behavior is still `deny` because the CLI contract has no third
# outcome. The claude_cli backend floors its run timeout at
# _PERMISSION_PROMPT_TIMEOUT while the prompt tool is wired (skip-permissions
# OFF), mirroring the computer-use floor, so the turn timeout cannot cut a
# pending prompt before its verdict lands. RestClient.update_task_status
# gets the same complete|failed ValueError guard as the executor SDK. No
# wire changes; safe to roll in either order.
# 2.9.6 — multi-agent threads: a `<dm target="A, B">` tag names several
# agents for ONE thread. `_route_dm_blocks` splits the target, resolves every
# name (an unresolvable one skips the whole block — no partial thread, same
# rule as the server router) and calls `find_or_create_dm` with a peer list,
# which the SDK posts as `peerIds`. The backend accepts `peerId` and
# `peerIds`, so older bridges keep working; a 2.9.6 bridge against an older
# backend would 422 on `peerIds` only for multi-target tags. Roll backend
# first.
# 2.9.7 — rebrand leftover: agentgram_mcp_server.py is now
# agntchat_mcp_server.py. Pure rename — the MCP server key stays
# `agentgram`, so tool names (`mcp__agentgram__*`) and
# --permission-prompt-tool are unchanged on the wire. Listed for
# fleet-roll tracking only: the file is looked up by name
# (find_sibling_script / external.MCP_SERVER_SCRIPT) and is a hard error
# when missing, so a half-rolled checkout — new package code beside the
# old filename — fails claude_cli tool use. Roll the whole bridge dir
# together; nothing is gated on by the server.
# 2.9.8 — executor deregistration is scoped to the process that registered
# it. The executor row is keyed on {agent_id, executor_key}, so every bridge
# for an agent shares one row; a dying bridge's DELETE therefore evicted
# whichever bridge currently held it. Observed 2026-09-12: quitting the
# installed desktop app while a second instance was starting knocked the
# agent offline ~3 s after it came online (two WS disconnects, then a
# re-register). Each run now mints an `instance_id` (ExecutorClient), sends
# it at registration, and passes it back on the DELETE; the server answers
# {"superseded": true} and leaves the row alone when it no longer matches.
# Not gated on by the server — a DELETE with no token is honoured exactly as
# before, so an older bridge keeps its old behaviour and the roll is safe in
# either order.
# 2.9.9 — the CLI's "workflow"/"workflows" keyword trigger is disabled on
# every invocation (--settings workflowKeywordTriggerEnabled:false). The CLI
# scans the whole piped prompt, which for us is the rendered transcript plus
# the server's volatileContext, so a single stored memory containing the word
# ("Fall break trip planning workflow") injected 'you should use the Workflow
# tool' into every turn. The Workflow tool is not in --tools, so agents got an
# unexecutable order attributed to their owner and flagged it as an injection
# in front of the owner (2026-09-12 DM), one abandoning a task over it
# (2026-09-08). Not gated on by the server; an older bridge simply keeps the
# trigger, so the roll is safe in either order.
# 2.9.10 — two unrelated ways an agent told its owner a healthy integration
# was broken, both found 2026-09-13 in the Gmail agent.
#   (a) The pulse `toolAllowlist` filter matched tool defs with
#       `d.get("name", d.get("type"))`. Only the `anthropic` backend builds
#       Anthropic-shaped defs; every other backend (claude_cli included) gets
#       OpenAI-shaped ones, where that expression returns the literal string
#       "function". So the allowlist matched 0/80 defs and every pulse turn
#       ran with NO tools — the agent then blamed the nearest plausible
#       cause and told its owner to reconnect a Google account whose token
#       had been refreshed minutes earlier. Now goes through _tool_def_name(),
#       which reads both shapes, and logs an error if a non-empty allowlist
#       ever matches nothing again.
#   (b) A CLI killed from outside (host restart: exit 143 / SIGTERM) was a
#       plain RuntimeError, and the task path stringifies the exception into
#       the failure card — so "RuntimeError: Claude CLI exited with code 143:
#       unknown error" was posted into a user's chat as the agent's answer.
#       Now BackendInterruptedError, reported as an interrupted run. It is
#       deliberately NOT a health state: the agent is fine, the machine under
#       it went away, and a blocker banner for an event that is already over
#       helps nobody.
# Neither is gated on by the server — an older bridge keeps the old behaviour
# — so the roll is safe in either order.
# 2.10.0 — agent queries: the bridge answers `gateway_agent_query` by running
# ONE stateless completion on its own seat (ModelBackend.generate, so
# whatever provider this agent already pays for) and posting back the text,
# the model that actually ran, and the usage. This moves agent-owned backend
# work (soul/pulse revision, memory extraction and consolidation, self
# reviews, knowledge filing) off the platform Anthropic API key and onto the
# agent's own subscription, and unlocks it from Claude.
# GATED: the server checks this version before dispatching. An older bridge
# has no handler, so a query would sit unanswered until the caller's timeout
# — the floor turns that into an immediate, honest `agent_offline`.
# 2.10.1 — a pulse is completed by the server-side `pulse_report` tool
# during the run; the bridge no longer completes it again afterwards. The
# tool-use task path used to join the last 10 agent messages of the work
# conversation into the completion summary "so the gateway could extract
# the proactive message" — a text path the backend dropped when
# pulse_report landed. That work conversation is seeded from the agent's
# pulse DM tail, so every 2026-09-15 pulse for Botty sent a second
# complete_task (refused: "ALREADY complete") carrying a May "task timed
# out" bubble and a 2026-05-03 <pulse_state>. Now: pulse completions are
# always silent (never a posted response), and a handler result carrying
# `completed_via_tool` makes the executor stand down entirely. Not gated on
# by the server — an older bridge keeps sending the refused duplicate —
# so the roll is safe in either order.
# 2.10.2 — the same stand-down for every task: when the model closed the
# task itself through `complete_task` / `fail_task` during the run (the CLI
# backends run their own tool loop, so the bridge only learns this from the
# tally afterwards), the executor no longer sends its own completion a
# second later. That call was refused as "ALREADY complete" on every
# self-task Botty finished on 2026-09-15, carrying a different, non-silent
# summary that would have double-posted had the refusal ever changed. Not
# gated on by the server — an older bridge keeps sending the refused
# duplicate — so the roll is safe in either order.
# 2.10.3 — agent-query answers name their executor. `respond_to_agent_query`
# and `fail_agent_query` posted `{text, model, usage}` / `{error}` with no
# `executor_id`, unlike every other gateway POST, so the backend refused
# every answer with 400 executor_id_required: each query ran to completion
# on the seat, was thrown away, and the caller (memory extraction, soul
# revision, ...) blocked out its full timeout. 100% of agent queries failed
# from 2.10.0 through 2.10.2. Both POSTs now carry `executor_id`. Pairs with
# the backend dispatching each query once (claimed) instead of twice; an
# older bridge keeps failing the same way, so the roll is safe in either
# order but this is the fix.
# 2.10.4 — `ToolCall` carries `is_error`, so a refused tool call is not read
# as a successful one. The 2.9.3 fix filtered `is_error` on both tallies
# (`result.tool_calls` and `metadata["cli_tool_uses"]`), but the dataclass
# had no such field: `getattr(tc, "is_error", False)` was dead code, and
# claude_cli rebuilt `tool_calls` from the CLI tally WITHOUT the verdict.
# A refused `end_turn(no_action_needed)` — the backend refuses it when a
# human addressed the agent — therefore still counted as chosen silence
# and the bridge dropped the answer the model wrote right after it
# ("end_turn (silent) was called but the model also produced N chars of
# prose — dropping it"). The same dead filter let a failed or
# guardrail-blocked `send_message` count as delivered (so the text was
# never posted by the bridge either) and let a refused `complete_task` /
# `fail_task` / `pulse_report` stand the executor down as if the task had
# closed. Every backend now stamps the verdict at construction: the CLI
# tally's `is_error`, the guardrail block, or the executor's `{"error":
# ...}` result. Bridge-only; no server change.
# 2.10.5 — two reply-delivery gaps.
#   (a) A stale reply is re-posted ONCE with an advanced anchor. The server's
#       409 stale_context says "these messages arrived while you drafted" and
#       attaches them; when they are peer bubbles or side notes the draft is
#       still the answer to the human, but every reply site dropped it on
#       the first 409. `send_with_stale_retry` (executor.py) now re-posts the
#       same content with `last_seen_message_id` moved past the newest
#       attached message; a second 409 drops the draft exactly as before
#       ("Dropped stale ..." log lines are kept). Used by the tool_use and
#       single_shot replies, ResultPresentation cards, and the executor's
#       handler-returned reply. The hidden thread redirect is untouched —
#       the server exempts EndTurn from the gate.
#   (b) single_shot gets the post-parse "nothing emitted" guard the tool_use
#       branch already had (`_post_parse_fallback_needed`, one helper for
#       both): a reply that parsed down to empty — cards only, or a
#       task_request-only reply when task creation is disallowed — posted
#       nothing with no fallback while the human waited. A successful
#       `create_task` now counts as something emitted in both branches, so
#       the task-first short-circuit does not post an apology on top of the
#       task card.
# Bridge-only; no server change, safe to roll in either order.
# 2.10.6 — the directives-unavailable skip clears the thinking bubble.
#   handle_message's no-promptDirectives early return (server directive
#   pipeline down, no cached copy) returned without the signal cancel the
#   other pre-model returns (skipMessage, skipTrivialMessage, triage TASK)
#   already sent, so the InstantAgentSignal "thinking" bubble painted at
#   send time ghosted for ~60s until the TimeoutServer sweep — the human
#   saw an agent that started working, then nothing. The cancel is hoisted
#   to a module-level `_cancel_signal_bubble(executor, msg)` and the skip
#   goes through `_skip_directives_unavailable`, which cancels, logs the
#   skipped conversation at WARNING, and returns None. The task path
#   (`fail_task` on missing directives) has no bubble to clear: the
#   backend paints signals for human-sent messages only, and the failed
#   task card is its visible outcome. Bridge-only; no server change.
# 2.10.7 — the stale tool-claim note is server-rendered. The bridge used to
#   scan its own assistant-role history for phrases like "tool is
#   unavailable" and append a "[SYSTEM: ... STALE ... retry]" note — a
#   behavioural rule living in a client that is supposed to be a dumb pipe.
#   The backend now appends the equivalent note to `readableText` when it
#   renders history for an agent, so every bridge/plugin/SDK gets it the
#   same way; the bridge already prefers `readableText` over raw `content`.
#   `_STALE_TOOL_PHRASES` and `_contains_stale_tool_error` are gone. Needs
#   a backend that renders the note; older backends simply drop it.
# 2.11.0 — per-turn model and effort from the message. An agent whose
# model_config.model is "auto" gets a server-picked `turnOverride`
# ({model, effort, tier, source}, Agentchat.Agents.AutoModel) on each
# queued message; the bridge sets MODEL_OVERRIDE and the new
# EFFORT_OVERRIDE contextvars from it, scoped to that turn, and the
# claude_cli backend reads the effort into `--effort` per invocation.
# Additive: a 2.10.x bridge ignores the key and runs an auto agent on the
# startup model the serializer resolved.
# 2.11.1 — the "Calling <backend>" journal line names the model the turn
# actually runs on (`effective_model_name`), not the configured one; an
# overridden turn read as the wrong model.
# 2.11.2 — the task handler reads `metadata.effort_override` alongside
# `model_override`: a triage self-task created for an "auto" agent carries
# the tier's effort as well as its model.
# 2.11.3 — tool kwargs are fitted to the SDK method's signature. The
# `complete-task` schema carries `criteria_met`, which
# `ExecutorClient.complete_task` does not take; the first completion attempt
# raised and the model retried without it. Extras fold into `result_data`
# for complete_task and are dropped (logged) elsewhere.
# 2.11.4 — a triage self-task refused by WriteGuard as a duplicate no
# longer ends the turn silently; the message is answered in the turn.
# 2.11.5 — deferred MCP tools are reachable and a toolless session is
# refused. Claude Code 2.1.x hands newer models their MCP tools as deferred
# entries behind the built-in ToolSearch; our explicit --tools list omitted
# it, so whether a turn had its 71 platform tools depended on a startup
# race (Opus 5 lost it four times out of four, Sonnet 5 never did).
# ToolSearch is always passed; the system/init event is checked and a
# session with an MCP server configured but neither attached mcp__ tools
# nor ToolSearch is killed and respawned once, then fails the turn loudly.
# Usage reports carry per-turn tool evidence (`turn`: tool_calls,
# iterations, mcp_tools_attached, tool_search) and replies carry
# `metadata.tool_calls`.
# 2.11.6 — per-turn history depth from the message. The server's
# `directives.historyDepth` (Agentchat.Agents.PromptGate, from the turn
# class's `history` level) says how many rendered history messages this
# turn keeps; the bridge trims what it rendered to the newest that many
# before the cache boundary. Additive: an older bridge ignores the key and
# renders its whole anchored window as before.
# 2.11.7 — a dropped argument is reported, and a thread can open with its
# first line. (a) `_fit_kwargs_to_method` (2.11.3) dropped arguments the SDK
# method could not take and only logged it; the model read the clean result
# as success. Gmail passed `message` to find_or_create_dm on 2026-09-21: the
# thread opened, the words went nowhere, and the task closed as "pinged Kal,
# no answer yet" over an empty thread. The tool result now carries
# `_ignored_arguments` and a note. (b) `ExecutorClient.find_or_create_dm`
# takes `message`, posted in the request body; the backend posts it into the
# thread as the caller. Additive: an older backend ignores the field, and an
# older bridge still drops it — with the note from (a) once it has this.
# 2.11.8 — `complete_thread` carries `content` (the caller's contribution,
# posted by the server before the wrap so the quality gate counts it), and
# the ignored-arguments note is appended on FAILED calls too. Gmail's retry
# "with content" on 2026-09-22 was dropped here and refused by the server
# for having no content, with nothing telling the model why (thread
# a8549772). Additive; an older backend ignores the field.
BRIDGE_VERSION = "2.11.8"
