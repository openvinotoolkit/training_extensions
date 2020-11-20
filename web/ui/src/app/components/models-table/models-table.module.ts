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
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';
import {MatTableModule} from '@angular/material/table';
import {MatSortModule} from '@angular/material/sort';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatMenuModule} from '@angular/material/menu';
import {MatIconModule} from '@angular/material/icon';
import {IdlpModelMenuModule} from '@idlp/components/model-menu/model-menu.module';
import {IdlpModelsTableComponent} from './models-table.component';
import {IdlpModelTableRowModule} from '@idlp/components/model-table-row/model-table-row.module';


@NgModule({
  declarations: [
    IdlpModelsTableComponent
  ],
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatInputModule,
    MatTableModule,
    MatSortModule,
    MatMenuModule,
    MatProgressBarModule,
    MatIconModule,
    IdlpModelMenuModule,
    IdlpModelTableRowModule
  ],
  exports: [
    IdlpModelsTableComponent
  ]
})
export class IdlpModelsTableModule {
}
