/**
 * Generate the TypeScript contract types from the engine's language-neutral JSON Schema
 * (docs/contracts/*.schema.json — the single source of truth, emitted by `corpus-studio
 * platform-schemas`). Run: `npm run gen:contracts`. The output is committed so the build is
 * deterministic; CI regenerates and diffs to catch drift from the engine contracts.
 *
 * Each schema compiles to its own module (its nested $defs stay module-local) so shared nested
 * types like Ref / MemoryMetrics can't collide across files; the index re-exports each root type.
 */
import { compileFromFile } from "json-schema-to-typescript";
import { readdirSync, writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const schemaDir = join(here, "..", "..", "..", "docs", "contracts");
const outDir = join(here, "..", "src", "contracts");
mkdirSync(outDir, { recursive: true });

const files = readdirSync(schemaDir).filter((f) => f.endsWith(".schema.json")).sort();
const roots = [];

for (const file of files) {
  const schema = JSON.parse(readFileSync(join(schemaDir, file), "utf-8"));
  const rootName = schema.title || file.replace(".schema.json", "");
  const ts = await compileFromFile(join(schemaDir, file), {
    bannerComment: `/* GENERATED from docs/contracts/${file} — do not edit. Run: npm run gen:contracts */`,
    additionalProperties: false,
    declareExternallyReferenced: true,
    enableConstEnums: false,
    style: { singleQuote: false, semi: true },
  });
  writeFileSync(join(outDir, `${rootName}.ts`), ts, "utf-8");
  roots.push(rootName);
}

const index =
  `/* GENERATED — do not edit. Run: npm run gen:contracts */\n` +
  roots.map((n) => `export type { ${n} } from "./${n}";`).join("\n") +
  "\n";
writeFileSync(join(outDir, "index.ts"), index, "utf-8");

console.log(`Generated ${roots.length} contract type modules: ${roots.join(", ")}`);
