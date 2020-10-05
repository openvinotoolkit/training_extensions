/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {IdlpScatterChartComponent} from './scatter-chart.component';
import {PlotlyModule} from "angular-plotly.js";
import {MatFormFieldModule} from "@angular/material/form-field";
import {MatSelectModule} from "@angular/material/select";
import {FlexModule} from "@angular/flex-layout";
import {ReactiveFormsModule} from "@angular/forms";


@NgModule({
  declarations: [
    IdlpScatterChartComponent
  ],
  exports: [
    IdlpScatterChartComponent
  ],
  imports: [
    CommonModule,
    PlotlyModule,
    MatFormFieldModule,
    MatSelectModule,
    FlexModule,
    ReactiveFormsModule
  ]
})
export class IdlpScatterChartModule {
}
