import { readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { marked } from "marked";

// Astro evaluates this module during the site build from the `site` directory.
// Keeping the source root relative to cwd lets the generated pages stay tied to
// the repository Markdown instead of duplicating documentation in the site.
const repoRoot = join(process.cwd(), "..");

type DocDefinition = {
  slug: string;
  title: string;
  source: string;
};

const titleFromMarkdown = (markdown: string, fallback: string) =>
  markdown.match(/^#\s+(.+)$/m)?.[1]?.replace(/[`*_]/g, "") ?? fallback;

const sourceDocs: DocDefinition[] = [
  { slug: "index", title: "Documentation index", source: join(repoRoot, "docs/INDEX.md") },
  { slug: "math", title: "Mathematical background", source: join(repoRoot, "docs/MATH.md") },
  { slug: "architecture", title: "Architecture", source: join(repoRoot, "docs/ARCHITECTURE.md") },
  { slug: "api-reference", title: "API reference", source: join(repoRoot, "docs/API_REFERENCE.md") },
  { slug: "usage", title: "Usage guide", source: join(repoRoot, "docs/USAGE.md") },
  { slug: "developer-guide", title: "Developer guide", source: join(repoRoot, "docs/DEVELOPER_GUIDE.md") },
  { slug: "engineering-principles", title: "Engineering principles", source: join(repoRoot, "docs/ENGINEERING_PRINCIPLES.md") },
  { slug: "deployment", title: "Deployment", source: join(repoRoot, "docs/DEPLOYMENT.md") },
  { slug: "extensions", title: "Extension guide", source: join(repoRoot, "docs/EXTENSIONS.md") },
  { slug: "api-migration", title: "v1 to v2 migration", source: join(repoRoot, "docs/API_MIGRATION.md") },
  { slug: "contributing", title: "Contributing", source: join(repoRoot, "CONTRIBUTING.md") },
];

export const docs = sourceDocs.map((doc) => {
  const markdown = readFileSync(doc.source, "utf8");
  return { ...doc, title: titleFromMarkdown(markdown, doc.title), markdown };
});

export const getDoc = (slug: string) => docs.find((doc) => doc.slug === slug);

const slugForSource = (source: string) => docs.find((doc) => doc.source === source)?.slug;

export const renderMarkdown = (markdown: string) => {
  const rewritten = markdown.replace(/\]\(([^)#]+\.md)(#[^)]+)?\)/gi, (_match, target: string, anchor = "") => {
    const source = join(repoRoot, target);
    const slug = slugForSource(source) ?? slugForSource(join(repoRoot, "docs", target));
    return slug ? `](/driftflow/docs/${slug}/${anchor})` : `](${target}${anchor})`;
  });
  return marked.parse(rewritten, { gfm: true, breaks: false });
};

export const relativeSource = (source: string) => relative(repoRoot, source);
