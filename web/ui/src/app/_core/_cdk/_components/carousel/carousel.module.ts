/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {IdlpCdkCarouselComponent} from './carousel.component';
import {MatIconModule} from "@angular/material/icon";
import {FlexModule} from "@angular/flex-layout";


@NgModule({
  declarations: [
    IdlpCdkCarouselComponent
  ],
  imports: [
    CommonModule,
    MatIconModule,
    FlexModule
  ],
  exports: [
    IdlpCdkCarouselComponent
  ]
})
export class IdlpCdkCarouselModule {
}
