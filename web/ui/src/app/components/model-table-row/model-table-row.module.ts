import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';
import {MatTableModule} from '@angular/material/table';
import {MatSortModule} from '@angular/material/sort';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatIconModule} from '@angular/material/icon';
import {IdlpModelMenuModule} from '@idlp/components/model-menu/model-menu.module';
import {IdlpModelTableRowComponent} from '@idlp/components/model-table-row/model-table-row.component';


@NgModule({
  declarations: [
    IdlpModelTableRowComponent
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
    IdlpModelMenuModule
  ],
  exports: [
    IdlpModelTableRowComponent
  ]
})
export class IdlpModelTableRowModule {
}
