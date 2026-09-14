# driftflow · site

The marketing site for the [driftflow](https://github.com/sachncs/driftflow)
Python library. Built with [Astro](https://astro.build) and Tailwind CSS,
deployed to GitHub Pages from the `dist/` output of this folder.

## Develop

```bash
cd site
npm install
npm run dev      # http://localhost:4321/driftflow
```

## Build

```bash
npm run build    # outputs to ./dist
npm run preview   # serve the build locally
```

## Deploy

GitHub Actions builds `site/` and publishes `dist/` to the `gh-pages` branch
on every push to `master`. The workflow lives at
`.github/workflows/pages.yml` in the repository root.

## Conventions

- Content is hand-authored in `src/pages/` and `src/components/`. Nothing is
  rendered from `README.md` or `docs/`.
- All design tokens live in `src/styles/global.css` and `tailwind.config.mjs`.
- Public assets (logo, favicon, OG image) live in `public/`.
- The site is served from the `/driftflow` base path because GitHub Pages
  publishes project sites under `https://<owner>.github.io/<repo>/`.