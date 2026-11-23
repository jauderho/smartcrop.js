import js from '@eslint/js';
import globals from 'globals';

export default [
  {
    ignores: [
      'node_modules/',
      'examples/*.min.js',
      'examples/*-min.js',
      'examples/face-api.js',
      'examples/jquery.js',
      'examples/q.js',
      'examples/underscore.js',
      'examples/face-api/',
    ],
  },
  js.configs.recommended,
  {
    files: ['**/*.js'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'script',
      globals: {
        ...globals.browser,
        ...globals.node,
        define: 'readonly',
      },
    },
  },
  {
    files: ['test/*.js', 'examples/*.js'],
    rules: { 'no-console': 'off' },
    languageOptions: {
      globals: {
        smartcrop: 'readonly',
        Benchmark: 'readonly',
        $: 'readonly',
        debugDraw: 'readonly',
        _: 'readonly',
        cv: 'readonly',
        tracking: 'readonly',
        faceapi: 'readonly',
      },
    },
  },
  {
    // debugDraw is defined here; don't flag it as a redeclared global.
    files: ['examples/smartcrop-debug.js'],
    languageOptions: { globals: { debugDraw: 'off' } },
  },
  {
    // The published library stays ES5 so it keeps working in old browsers.
    files: ['smartcrop.js'],
    languageOptions: {
      ecmaVersion: 5,
      globals: { Promise: 'readonly', Uint8ClampedArray: 'readonly' },
    },
  },
];
