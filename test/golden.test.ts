import { expect, it } from 'bun:test';
import smartcrop from '../src/smartcrop';
import golden from './fixtures/golden-crops.json';
import { computeGolden } from './golden';

// Crops for every sample image must match the results recorded from
// smartcrop 2.0.5. Regenerate the fixture only for intended changes.
it('matches the recorded crops for every sample image', async () => {
  expect(await computeGolden(smartcrop)).toEqual(golden);
}, 60_000);
