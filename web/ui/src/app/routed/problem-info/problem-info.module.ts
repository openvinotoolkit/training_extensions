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

import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FlexModule} from '@angular/flex-layout';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {IdlpCdkCarouselModule} from '@idlp/cdk/_components/carousel/carousel.module';
import {IdlpFineTuneDialogModule} from '@idlp/components/fine-tune-dialog/fine-tune-dialog.module';
import {IdlpProblemInfoComponent} from './problem-info.component';
import {IdlpScatterChartModule} from "@idlp/components/scatter-chart/scatter-chart.module";
import {IdlpModelsTableModule} from "@idlp/components/models-table/models-table.module";
import {IdlpModelMenuModule} from "@idlp/components/model-menu/model-menu.module";
import {MatFormFieldModule} from "@angular/material/form-field";
import {MatSelectModule} from "@angular/material/select";
import {HttpClientModule} from "@angular/common/http";
import {MatAutocompleteModule} from "@angular/material/autocomplete";
import {MatInputModule} from "@angular/material/input";

@NgModule({
  declarations: [
    IdlpProblemInfoComponent
  ],
  imports: [
    HttpClientModule,
    CommonModule,
    FlexModule,
    MatButtonModule,
    MatIconModule,
    MatDialogModule,
    IdlpCdkCarouselModule,
    IdlpFineTuneDialogModule,
    IdlpScatterChartModule,
    IdlpModelsTableModule,
    IdlpModelMenuModule,
    MatFormFieldModule,
    MatSelectModule,
    MatAutocompleteModule,
    MatInputModule
  ],
  exports: [
    IdlpProblemInfoComponent
  ]
})
export class IdlpProblemInfoModule {
}
