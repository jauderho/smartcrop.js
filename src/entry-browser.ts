import smartcrop from './smartcrop';

const g = globalThis as Record<string, unknown>;
g.smartcrop = g.SmartCrop = smartcrop;
