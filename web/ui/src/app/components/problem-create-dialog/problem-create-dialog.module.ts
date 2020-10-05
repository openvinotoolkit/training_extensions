/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FlexModule} from '@angular/flex-layout';
import {FormsModule, ReactiveFormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule} from '@angular/material/dialog';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';
import {MatToolbarModule} from '@angular/material/toolbar';
import {IdlpProblemCreateDialogComponent} from "./problem-create-dialog.component";
import {MatProgressBarModule} from "@angular/material/progress-bar";
import {MatIconModule} from "@angular/material/icon";
import {IdlpDisplayFieldModule} from "@idlp/cdk/_components/display-field/display-field.module";


@NgModule({
  declarations: [
    IdlpProblemCreateDialogComponent
  ],
  imports: [
    CommonModule,
    FlexModule,
    FormsModule,
    ReactiveFormsModule,
    MatToolbarModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatProgressBarModule,
    MatIconModule,
    IdlpDisplayFieldModule,
  ],
  exports: [
    IdlpProblemCreateDialogComponent
  ],
  providers: []
})
export class IdlpProblemCreateDialogModule {
}
