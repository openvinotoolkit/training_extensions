/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {IdlpProblemAssetsComponent} from '@idlp/routed/problem-assets/problem-assets.component';
import {IdlpProblemInfoComponent} from '@idlp/routed/problem-info/problem-info.component';
import {IdlpProblemListComponent} from '@idlp/routed/problem-list/problem-list.component';

const routes: Routes = [
  {
    path: '',
    children: [
      {
        path: '',
        component: IdlpProblemListComponent
      },
      {
        path: ':id',
        redirectTo: ':id/info'
      },
      {
        path: ':id',
        children: [
          {
            path: 'info',
            component: IdlpProblemInfoComponent
          },
          {
            path: 'assets',
            children: [
              {
                path: '',
                component: IdlpProblemAssetsComponent

              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '**',
    redirectTo: ''
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class RoutingModule {
}
