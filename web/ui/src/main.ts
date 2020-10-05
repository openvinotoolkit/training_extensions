/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import { enableProdMode } from '@angular/core';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { environment } from '@environments/environment';
import 'hammerjs';
import { IdlpRootModule } from './app/idlp-root.module';

if (environment.production) {
  enableProdMode();
}

platformBrowserDynamic().bootstrapModule(IdlpRootModule)
  .catch(err => console.error(err));
