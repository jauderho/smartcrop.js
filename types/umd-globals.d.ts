// Globals smartcrop.js probes for in its UMD wrapper. Used for type checking only.
declare var define: ((factory: () => unknown) => void) & { amd?: unknown };

interface Window {
  smartcrop: unknown;
  SmartCrop: unknown;
}
