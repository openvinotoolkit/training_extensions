/**
 * Copyright (c) 2020 Intel Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
