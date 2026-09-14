// @ts-check
import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// driftflow is published under sachncs.github.io/driftflow as a project site.
// `site` is used to produce absolute URLs for OG tags and structured data.
export default defineConfig({
  site: "https://sachncs.github.io",
  base: "/driftflow",
  trailingSlash: "ignore",
  build: {
    assets: "assets",
    inlineStylesheets: "auto",
  },
  integrations: [
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  vite: {
    build: {
      cssCodeSplit: true,
    },
  },
  prefetch: {
    prefetchAll: false,
    defaultStrategy: "hover",
  },
});