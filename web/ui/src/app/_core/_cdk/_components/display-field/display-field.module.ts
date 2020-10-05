/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FlexModule } from '@angular/flex-layout';
import { IdlpDisplayFieldComponent } from './display-field.component';


@NgModule({
  declarations: [
    IdlpDisplayFieldComponent
  ],
  imports: [
    CommonModule,
    FlexModule
  ],
  exports: [
    IdlpDisplayFieldComponent
  ]
})
export class IdlpDisplayFieldModule {
}
