// Aggregates every chapter's diagram module into one id -> fn map.
import { diagrams as preliminaries } from './preliminaries.mjs';
import { diagrams as ndarray } from './ndarray.mjs';
import { diagrams as linearAlgebra } from './linear-algebra.mjs';
import { diagrams as calculus } from './calculus.mjs';
import { diagrams as autograd } from './autograd.mjs';
import { diagrams as pandas } from './pandas.mjs';
import { diagrams as probability } from './probability.mjs';
import { diagrams as lookupApi } from './lookup-api.mjs';
import { diagrams as ch02LinearRegression } from './ch02-linear-regression.mjs';
import { diagrams as ch03LinearClassification } from './ch03-linear-classification.mjs';
import { diagrams as ch04Mlp } from './ch04-mlp.mjs';
import { diagrams as ch05BuildersGuide } from './ch05-builders-guide.mjs';
import { diagrams as ch06Cnn } from './ch06-cnn.mjs';
import { diagrams as ch07ModernCnn } from './ch07-modern-cnn.mjs';
import { diagrams as ch08Rnn } from './ch08-rnn.mjs';
import { diagrams as ch09Optimization } from './ch09-optimization.mjs';
import { diagrams as ch10Attention } from './ch10-attention.mjs';
import { diagrams as ch11Transformers } from './ch11-transformers.mjs';
import { diagrams as ch12ModernRnn } from './ch12-modern-rnn.mjs';
import { diagrams as ch13Performance } from './ch13-performance.mjs';
import { diagrams as ch14Rl } from './ch14-rl.mjs';
import { diagrams as ch15DeepRl } from './ch15-deep-rl.mjs';
import { diagrams as ch16Gans } from './ch16-gans.mjs';
import { diagrams as ch17Diffusion } from './ch17-diffusion.mjs';
import { diagrams as ch33MdlLa } from './ch33-mdl-la.mjs';
import { diagrams as ch34MdlCalc } from './ch34-mdl-calc.mjs';
import { diagrams as ch35MdlOpt } from './ch35-mdl-opt.mjs';
import { diagrams as ch36MdlProb } from './ch36-mdl-prob.mjs';
import { diagrams as ch37MdlInfo } from './ch37-mdl-info.mjs';
import { diagrams as ch38MdlDyn } from './ch38-mdl-dyn.mjs';
import { diagrams as ch39Tools } from './ch39-tools.mjs';

export const diagrams = {
  ...preliminaries,
  ...ndarray,
  ...linearAlgebra,
  ...calculus,
  ...autograd,
  ...pandas,
  ...probability,
  ...lookupApi,
  ...ch02LinearRegression,
  ...ch03LinearClassification,
  ...ch04Mlp,
  ...ch05BuildersGuide,
  ...ch06Cnn,
  ...ch07ModernCnn,
  ...ch08Rnn,
  ...ch09Optimization,
  ...ch10Attention,
  ...ch11Transformers,
  ...ch12ModernRnn,
  ...ch13Performance,
  ...ch14Rl,
  ...ch15DeepRl,
  ...ch16Gans,
  ...ch17Diffusion,
  ...ch33MdlLa,
  ...ch34MdlCalc,
  ...ch35MdlOpt,
  ...ch36MdlProb,
  ...ch37MdlInfo,
  ...ch38MdlDyn,
  ...ch39Tools,
};
