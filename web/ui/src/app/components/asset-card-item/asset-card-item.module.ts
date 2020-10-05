/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FlexModule} from '@angular/flex-layout';
import {MatButtonModule} from '@angular/material/button';
import {MatCardModule} from '@angular/material/card';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {RouterModule} from '@angular/router';
import {PipesModule} from '@idlp/cdk/_pipes/pipes.module';
import {IdlpAssetCardItemComponent} from './asset-card-item.component';
import {MatCheckboxModule} from "@angular/material/checkbox";
import {FormsModule} from "@angular/forms";
import {MatProgressSpinnerModule} from "@angular/material/progress-spinner";


@NgModule({
  declarations: [
    IdlpAssetCardItemComponent
  ],
  imports: [
    CommonModule,
    RouterModule,
    FlexModule,
    PipesModule,
    MatProgressBarModule,
    MatTooltipModule,
    MatButtonModule,
    MatIconModule,
    MatCardModule,
    MatCheckboxModule,
    FormsModule,
    MatProgressSpinnerModule
  ],
  exports: [
    IdlpAssetCardItemComponent
  ]
})
export class IdlpAssetCardItemModule {
}
