// Aggregates every chapter's diagram module into one id → fn map.
// To add a chapter: create <file>.mjs exporting `diagrams`, import it
// here, and spread it in. Ids are global, so keep the `<file>-<concept>`
// prefix convention to avoid collisions.

import { diagrams as ndarray } from './ndarray.mjs';
import { diagrams as linearAlgebra } from './linear-algebra.mjs';
import { diagrams as calculus } from './calculus.mjs';
import { diagrams as autograd } from './autograd.mjs';
import { diagrams as pandas } from './pandas.mjs';
import { diagrams as probability } from './probability.mjs';

export const diagrams = {
  ...ndarray,
  ...linearAlgebra,
  ...calculus,
  ...autograd,
  ...pandas,
  ...probability,
};
