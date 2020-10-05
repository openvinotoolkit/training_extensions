/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {IdlpProblemCardItemComponent} from '@idlp/components/problem-card-item/problem-card-item.component';
import {IdlpCdkCarouselModule} from '@idlp/cdk/_components/carousel/carousel.module';
import {MatIconModule} from '@angular/material/icon';
import {MatCardModule} from '@angular/material/card';
import {MatButtonModule} from '@angular/material/button';
import {FlexModule} from "@angular/flex-layout";


@NgModule({
  declarations: [
    IdlpProblemCardItemComponent
  ],
  imports: [
    CommonModule,
    MatIconModule,
    MatButtonModule,
    MatCardModule,
    IdlpCdkCarouselModule,
    FlexModule
  ],
  exports: [
    IdlpProblemCardItemComponent
  ]
})
export class IdlpProblemCardItemModule {
}
