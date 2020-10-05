/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FlexModule} from '@angular/flex-layout';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatSelectModule} from '@angular/material/select';
import {IdlpDisplayFieldModule} from '@idlp/cdk/_components/display-field/display-field.module';
import {IdlpAssetCardItemModule} from '@idlp/components/asset-card-item/asset-card-item.module';
import {IdlpProblemAssetsComponent} from './problem-assets.component';
import {MatSnackBarModule} from "@angular/material/snack-bar";
import {MatTooltipModule} from "@angular/material/tooltip";

@NgModule({
  declarations: [
    IdlpProblemAssetsComponent
  ],
  imports: [
    CommonModule,
    FlexModule,
    IdlpAssetCardItemModule,
    MatButtonModule,
    MatIconModule,
    IdlpDisplayFieldModule,
    MatFormFieldModule,
    MatSelectModule,
    MatSnackBarModule,
    MatTooltipModule
  ],
  exports: [
    IdlpProblemAssetsComponent
  ]
})
export class IdlpProblemAssetsModule {
}
