/**
 * Generate the TypeScript contract types from the engine's language-neutral JSON Schema
 * (docs/contracts/*.schema.json — the single source of truth, emitted by `corpus-studio
 * platform-schemas`). Run: `npm run gen:contracts`. The output is committed so the build is
 * deterministic; CI regenerates and diffs to catch drift from the engine contracts.
 *
 * Each schema compiles to its own module (its nested $defs stay module-local) so shared nested
 * types like Ref / MemoryMetrics can't collide across files; the index re-exports each root type.
 */
import { compile } from "json-schema-to-typescript";
import { readdirSync, writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const schemaDir = join(here, "..", "..", "..", "docs", "contracts");
const outDir = join(here, "..", "src", "contracts");
mkdirSync(outDir, { recursive: true });

const files = readdirSync(schemaDir).filter((f) => f.endsWith(".schema.json")).sort();
const roots = [];

/**
 * json-schema-to-typescript treats an object schema carrying validation-only `if`/`then`
 * clauses in `allOf` as an opaque index signature, discarding the object's declared fields.
 * TypeScript cannot express those cross-field runtime conditions, so compile the structural
 * projection while leaving the checked-in JSON Schema (and its authoritative conditions)
 * untouched. Structural `allOf` members such as `$ref` intersections are retained.
 */
function structuralProjection(value) {
  if (Array.isArray(value)) {
    return value.map(structuralProjection);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  const projected = {};
  for (const [key, nested] of Object.entries(value)) {
    if (key === "allOf" && Array.isArray(nested)) {
      const structural = nested.filter(
        (clause) =>
          clause === null ||
          typeof clause !== "object" ||
          (!("if" in clause) && !("then" in clause) && !("else" in clause)),
      );
      if (structural.length > 0) {
        projected[key] = structuralProjection(structural);
      }
      continue;
    }
    projected[key] = structuralProjection(nested);
  }
  return projected;
}

for (const file of files) {
  const schema = JSON.parse(readFileSync(join(schemaDir, file), "utf-8"));
  const rootName = schema.title || file.replace(".schema.json", "");
  const ts = await compile(structuralProjection(schema), rootName, {
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
