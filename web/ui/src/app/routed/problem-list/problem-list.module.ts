/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FlexModule} from '@angular/flex-layout';
import {IdlpProblemCardItemModule} from '@idlp/components/problem-card-item/problem-card-item.module';
import {IdlpProblemListComponent} from '@idlp/routed/problem-list/problem-list.component';
import {IdlpProblemCreateDialogModule} from '@idlp/components/problem-create-dialog/problem-create-dialog.module';


@NgModule({
  declarations: [
    IdlpProblemListComponent
  ],
  imports: [
    CommonModule,
    FlexModule,
    IdlpProblemCardItemModule,
    IdlpProblemCreateDialogModule
  ],
  exports: [
    IdlpProblemListComponent
  ]
})
export class IdlpProblemListModule {
}
