# Driftflow Production Readiness, API Redesign, and Site Rebuild

> Status: implemented on branch `codex/driftflow-v2`; this document retains
> the original audit findings for traceability and the remaining checklist as
> release guidance.

## Summary

Perform a clean-break v2 redesign of the Python package and Astro site.

The current implementation passes its existing checks—178 tests, 100% line coverage, Ruff, formatting, mypy, demo smoke test, and Astro build—but the audit found significant reliability, API, dependency, documentation, and UX gaps.

## Complete issue inventory

### Python library

- Public API is built around loosely typed nested lists and unstructured `dict[str, list[float]]` results.
- No formal polymorphic contracts for drift models, schedules, solvers, or samplers.
- Solver registration uses mutable global dictionaries and process-wide state.
- Custom solver gamma resolution silently falls back to Heun behavior.
- `DatasetConfig(frozen=True)` still exposes a mutable `active_range` list.
- Configuration values are not validated for finiteness, positivity, ordering, or compatible ranges.
- Schedules accept invalid values and can produce negative, infinite, or NaN noise scales.
- `CosineSchedule` is described as cosine-based but implemented as a tangent approximation with a hard-coded clamp.
- `PolynomialSchedule` permits invalid exponents and unstable inputs.
- `sigmoid_decode_adjacency()` can overflow for large negative logits.
- Matrix helpers use non-strict `zip()` and can silently truncate malformed data.
- `matmul()` lacks dimension and empty-input validation.
- `SimpleGraphDenoiser.__call__()` does not validate feature/adjacency shapes against constructor dimensions.
- Drift outputs are not validated after every model evaluation.
- Noise schedule outputs are not validated before numerical integration.
- Solver outputs are not validated before becoming the next state.
- NaN and infinity may propagate into trajectories instead of producing actionable failures.
- `DVSSampler` is explicitly non-thread-safe and owns mutable RNG state.
- Reusing one sampler instance changes results implicitly because RNG state persists between calls.
- Sampling cancellation, step limits, and runaway-loop protection are absent.
- `terminal_time` is validated, but the relationship between `eps_bound`, `dt_min`, `dt_max`, and `dt_base` is not.
- No explicit graph invariant validation exists for square adjacency, symmetry, self-loops, or feature/node alignment.
- `info` uses scalar values wrapped in one-element lists.
- Public internal helpers are exposed inconsistently and lack a stable result schema.
- Logging is limited and there is no structured diagnostics or tracing hook.
- Approximation models are educational placeholders but are easy to mistake for production denoisers.
- Approximation model weights and fields remain mutable despite documentation implying stable/stateless behavior.
- The retired `make_drift_function()` adapter was removed; models now implement the typed `DriftModel` contract directly.
- The library has no benchmark/performance regression suite.
- Tests heavily target line coverage and happy-path behavior, but not contract/property/fuzz/performance behavior.
- No compatibility, migration, or versioned API policy exists.
- No explicit engineering principles document governs validation, immutability, error handling, or extension design.

### Tooling and CI

- `scripts/ci.sh` assumes a `python` executable that is absent in the local environment.
- CI does not run the Astro site build as part of the main quality gate.
- Frontend dependencies install with a fallback from `npm ci` to `npm install`, weakening lockfile reproducibility.
- `npm audit` reports 24 vulnerabilities: 11 moderate, 12 high, and 1 critical.
- The Astro dependency is directly affected by multiple security advisories.
- The unused Tailwind/sharp dependency stack was removed; the remaining Astro dependency tree is audited in CI.
- `pip-audit --strict` fails because the local editable package is not published/auditable.
- No SBOM, dependency policy, or vulnerability exception policy exists.
- No matrix verification exists for the actual supported Python versions in the local workflow.
- No test for package installation from wheel/sdist in isolated environments.
- No frontend accessibility, responsive, link, or interaction tests.
- No release smoke test against the published package.
- No pre-commit or unified quality command covering Python and site code.

### Documentation and consistency

- README and docs advertise production discipline while the implementation still contains educational approximations and weak runtime contracts.
- Several examples are not self-contained without undefined values or implied setup.
- The website’s hero code sample references undefined `features_0` and contains malformed displayed syntax.
- The website’s API sample advertises a custom solver signature that does not match the actual `SolverStep` contract.
- Website copy says “deterministic by default,” but deterministic output requires an explicit seed.
- Documentation, website, CI, and package metadata contain duplicated claims that can drift.
- The contributor guide claims Google-style conventions but does not enforce them across all artifacts.
- No explicit naming policy exists for semi-private members, extension hooks, or internal modules.
- No architecture decision records or engineering principles exist.
- No documented failure taxonomy or exception hierarchy exists.

### Website UX/UI

- Hero is visually dense and prioritizes marketing language over explaining the user journey.
- Primary action is ambiguous: install, GitHub, paper citation, and documentation compete equally.
- The first viewport shows a large code block before communicating a simple “what it does / when to use it” workflow.
- Dark purple/black palette has weak contrast in secondary text, pills, borders, and metadata.
- Decorative gradients, glow, glass surfaces, and marquee content create visual noise.
- The page has excessive vertical length for a single-purpose library landing page.
- Repeated marquee claims add motion and repetition without improving comprehension.
- Navigation has no mobile menu or equivalent compact navigation control.
- Site uses client-side copy/tab scripts without interaction tests or failure feedback.
- Copy failures are silently swallowed.
- Tab buttons lack complete ARIA state management (`aria-selected`, tab-panel relationships, keyboard behavior).
- Code samples are not consistently executable or copy-safe.
- No visible “documentation first” path for users who need API details.
- Metrics such as “100% coverage” and “production discipline” overstate confidence.
- Site provides no performance, bundle-size, accessibility, or responsive acceptance checks.
- Remote Google Fonts create an unnecessary runtime dependency and possible layout shift.
- Canonical/OG asset handling is not verified for the GitHub Pages base path.
- GitHub Pages workflow does not include a production link check or 404/asset verification.
- Color tokens are centralized in the site stylesheet and SVG assets; the site no longer uses Tailwind.
- The visual system lacks a clear semantic token layer for success, warning, error, interactive, and muted states.

## Implementation changes

### 1. Establish engineering foundations

Add:

- `docs/ENGINEERING_PRINCIPLES.md`
- `docs/ARCHITECTURE.md` rewrite
- `docs/API_MIGRATION.md`
- explicit Google Python style and public/private API boundaries
- exception taxonomy and runtime validation policy
- unified `scripts/check.sh` entrypoint
- pre-commit configuration
- versioned API policy for v2

Required conventions:

- public extension points use normal documented naming;
- semi-private underscore-prefixed APIs are not used as a convention;
- implementation details are isolated in private modules or classes rather than exposed as semi-private members;
- double-underscore name mangling is prohibited except for a demonstrated inheritance requirement;
- all public APIs use Google-style docstrings and complete annotations.

### 2. Redesign the Python domain model

Introduce typed domain objects:

- immutable `GraphState`
- immutable `DriftState`
- immutable `SampleResult`
- typed `StepDiagnostics`
- typed `SamplerDiagnostics`
- validated configuration objects

Introduce polymorphic interfaces:

- `DriftModel` abstract base class or protocol
- `NoiseSchedule` abstract base class
- `Solver` abstract base class
- `AdaptiveController` abstract base class
- `StateValidator` strategy
- `GraphDecoder` strategy

Implement concrete classes:

- `EulerMaruyamaSolver`
- `HeunSolver`
- `DvsAdaptiveController`
- validated linear, cosine, polynomial, and constant schedules
- typed approximation models explicitly marked as non-production

Replace global mutable solver registration with an instance-level `SolverRegistry` that:

- validates names and contracts;
- requires explicit controller/gamma metadata;
- prevents accidental collisions;
- supports isolated registries for tests and applications.

### 3. Make numerical behavior reliable

Add validation for:

- finite numeric values;
- positive and ordered timestep bounds;
- valid EMA/controller parameters;
- valid schedule parameters;
- graph shape invariants;
- drift output shape and finiteness;
- solver output shape and finiteness;
- noise scale finiteness and sign;
- maximum step count;
- terminal-time reachability.

Replace unstable numerical code:

- use a stable sigmoid implementation;
- use strict shape checks;
- reject malformed matrices instead of truncating;
- make schedule semantics explicit and mathematically correct;
- define behavior for zero noise, zero curvature, infinite inputs, and NaN inputs;
- ensure terminal-time handling cannot stall or overshoot.

Make sampling reproducibility explicit:

- sampler instances become stateless with respect to random state;
- accept an RNG or seed per `sample()` call;
- expose deterministic and nondeterministic modes clearly;
- document thread/process safety.

### 4. Redesign the result and error API

Replace the current list-based info dictionary with typed results:

- `SampleResult.features`
- `SampleResult.adjacency`
- `SampleResult.diagnostics`
- `StepDiagnostics` records
- explicit final time and step count fields

Add actionable exceptions:

- `ConfigurationError`
- `ShapeError`
- `NumericalStabilityError`
- `SolverContractError`
- `ScheduleError`
- `SamplingLimitError`

Document all failure modes and migration mappings from v1.

### 5. Expand verification

Add:

- property-based tests for shapes, finite values, monotonic time, and timestep bounds;
- fuzz tests for malformed matrices and extreme numeric inputs;
- contract tests for every solver, schedule, drift model, and decoder;
- reproducibility tests across repeated calls and isolated RNGs;
- concurrency tests;
- performance benchmarks for graph sizes;
- wheel/sdist installation tests;
- Python 3.10–3.14 matrix tests;
- mutation testing or equivalent quality signal;
- explicit tests proving educational approximations are labeled and isolated.

Coverage remains required, but acceptance shifts from line coverage alone to behavioral contracts.

### 6. Repair tooling and dependency security

- Replace `python` assumptions with `python3` or an activated environment.
- Make CI fail on dependency vulnerabilities above an agreed severity.
- Keep the static-site dependency tree minimal and audit it on every release.
- Remove `npm install` fallback from CI.
- Add lockfile integrity verification.
- Separate development-only tooling from deploy dependencies.
- Add package audit handling for unpublished local distributions.
- Generate an SBOM during release checks.
- Run Python and site checks together through one documented command.

### 7. Rebuild the website UX/UI

Adopt a clearer product narrative:

1. What Driftflow does.
2. Why adaptive timesteps matter.
3. Where it fits in a graph-diffusion pipeline.
4. Minimal install and executable example.
5. API and extension model.
6. Research caveat and production boundary.

Visual direction:

- reduce glow, blur, gradients, and decorative repetition;
- use a restrained neutral base with one strong accent and semantic status colors;
- improve contrast and text hierarchy;
- define all colors through semantic design tokens;
- use a compact responsive layout;
- add a real mobile navigation control;
- use one primary CTA: “Read the docs” or “Install”;
- make GitHub and paper links secondary;
- replace fake “live preview” claims with a real static diagram or honest label;
- distinguish “research reproduction” from “production-ready integration.”

Content fixes:

- replace malformed hero and API samples with tested, executable snippets;
- include complete variable definitions;
- update custom solver examples to the real v2 interface;
- remove unsupported claims about deterministic defaults, production solvers, and coverage guarantees;
- add a concise “limitations” section;
- link directly to API docs, migration docs, and examples.

Accessibility and interaction:

- semantic landmarks and headings;
- complete ARIA tab implementation;
- keyboard navigation;
- visible focus states;
- reduced-motion support;
- accessible contrast;
- copy buttons with success/error feedback;
- no silent interaction failures;
- local/self-hosted fonts or deliberate fallback strategy.

### 8. Website validation

Add:

- Astro build check;
- HTML/link checker;
- accessibility scan;
- responsive snapshots at desktop, tablet, and mobile widths;
- interaction tests for navigation, tabs, copy buttons, and anchors;
- asset/base-path verification for GitHub Pages;
- performance budget for CSS, JS, fonts, and images;
- production smoke test against the deployed URL.

## Test and acceptance criteria

The work is complete when:

- all v2 public APIs are typed, validated, documented, and covered by contract tests;
- malformed states fail with specific exceptions before numerical integration;
- repeated seeded samples are reproducible without hidden sampler state;
- custom solvers and schedules use explicit polymorphic contracts;
- no mutable global registry controls application behavior;
- Python checks pass across the supported version matrix;
- frontend dependencies have no unresolved critical/high production-relevant advisories;
- package, wheel, and sdist install successfully in clean environments;
- the site builds from a clean checkout with `npm ci`;
- the live page passes responsive, accessibility, interaction, and link checks;
- every displayed code sample is executable or explicitly labeled pseudocode;
- documentation, README, site copy, and API signatures agree;
- no claim of “production-ready” remains unless backed by an acceptance test.

## Assumptions and defaults

- The selected policy is a clean break: introduce v2 APIs and remove the current v1 extension contracts rather than preserving compatibility.
- The redesign does not use a semi-private underscore naming convention; public and private boundaries are explicit.
- The educational approximation models remain available, but are clearly namespaced and labeled as non-production.
- The package remains dependency-free at runtime unless a measured performance requirement justifies an optional backend.
- The site remains a static Astro/GitHub Pages deployment.
- No implementation files are changed during this planning pass.
