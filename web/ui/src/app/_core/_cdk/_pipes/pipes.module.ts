/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { DashedPipe } from '@idlp/cdk/_pipes/dashed.pipe';


@NgModule({
  declarations: [
    DashedPipe
  ],
  imports: [
    CommonModule
  ],
  exports: [
    DashedPipe
  ]
})
export class PipesModule {
}
