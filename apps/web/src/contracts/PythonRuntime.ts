/* GENERATED from docs/contracts/PythonRuntime.schema.json — do not edit. Run: npm run gen:contracts */

export type Architecture = string;
export type Compatible = boolean;
export type ContractVersion = "1.0.0";
export type Executable = string;
export type Implementation = string;
export type IncompatibilityReasons = string[];
export type IsVirtualEnvironment = boolean;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type Platform = string;
export type RuntimeId = string;
export type VenvAvailable = boolean;
export type Version = string;

/**
 * A discovered Python executable that can potentially create an isolated worker environment.
 *
 * Discovery never assumes the control-plane interpreter is the only installation. Compatibility is
 * an explicit verdict against the selected recipe, while ``venv_available`` proves the stdlib venv
 * module can be located without creating anything.
 */
export interface PythonRuntime {
  architecture?: Architecture;
  compatible?: Compatible;
  contract_version?: ContractVersion;
  executable: Executable;
  implementation?: Implementation;
  incompatibility_reasons?: IncompatibilityReasons;
  is_virtual_environment?: IsVirtualEnvironment;
  os?: OperatingSystem;
  platform?: Platform;
  runtime_id: RuntimeId;
  venv_available?: VenvAvailable;
  version?: Version;
}
