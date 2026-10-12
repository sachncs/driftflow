# driftflow site

Static Astro site for the driftflow v2 API.

## Develop

```bash
npm ci
npm run dev
```

The local site is served at `/driftflow` to match GitHub Pages.

## Build

```bash
npm run build
npm run preview
```

The production build is static and writes to `dist/`. Keep code examples,
navigation labels, and API claims synchronized with the Python package and
`docs/`.
