# Engineering Principles

1. **Make invalid states unrepresentable.** Use immutable validated domain
   objects at component boundaries.
2. **Prefer polymorphism to conditionals.** New models, schedules, controllers,
   solvers, and decoders implement contracts rather than modifying the core loop.
3. **Fail loudly and locally.** Raise typed exceptions at the boundary where a
   bad value is introduced.
4. **Keep state ownership explicit.** No hidden RNG state and no mutable global
   extension registry.
5. **Use Google Python style.** Public APIs have complete annotations and
   Google-style docstrings. Lines are limited to 100 characters.
6. **Avoid semi-private APIs.** Do not use underscore-prefixed members as a
   convention. Public extension points are documented; implementation details
   are isolated in private modules or concrete classes.
7. **Test behavior, not coverage numbers.** Contract, property, numerical,
   integration, and failure-path tests are required alongside coverage.
8. **Be honest about research status.** Educational approximations and claims
   of paper fidelity must be labeled clearly.
