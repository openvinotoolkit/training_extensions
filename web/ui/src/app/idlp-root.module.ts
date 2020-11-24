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
import {FlexModule} from '@angular/flex-layout';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatToolbarModule} from '@angular/material/toolbar';
import {BrowserModule} from '@angular/platform-browser';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {environment} from '@environments/environment';
import {WebsocketModule} from '@idlp/providers/websocket.module';
import {IdlpRootComponent} from '@idlp/root/idlp-root.component';
import {IdlpRootState} from '@idlp/root/idlp-root.state';
import {RoutingModule} from '@idlp/root/routing.module';
import {WsState} from '@idlp/root/ws.state';
import {IdlpProblemAssetsModule} from '@idlp/routed/problem-assets/problem-assets.module';
import {IdlpProblemInfoModule} from '@idlp/routed/problem-info/problem-info.module';
import {ProblemInfoState} from '@idlp/routed/problem-info/problem-info.state';
import {IdlpProblemListModule} from '@idlp/routed/problem-list/problem-list.module';
import {ProblemListState} from '@idlp/routed/problem-list/problem-list.state';
import {NgxsReduxDevtoolsPluginModule} from '@ngxs/devtools-plugin';
import {NgxsModule} from '@ngxs/store';
import {NgxsWebsocketPluginModule} from '@ngxs/websocket-plugin';
import {PlotlyModule} from 'angular-plotly.js';

import * as PlotlyJS from 'plotly.js/dist/plotly.js';
import {NgxsLoggerPluginModule} from '@ngxs/logger-plugin';

PlotlyModule.plotlyjs = PlotlyJS;

@NgModule({
  declarations: [
    IdlpRootComponent
  ],
  imports: [
    WebsocketModule.forRoot({
      url: environment.ws
    }),
    BrowserAnimationsModule,
    BrowserModule,
    RoutingModule,
    FlexModule,
    MatToolbarModule,
    MatButtonModule,
    MatIconModule,
    IdlpProblemListModule,
    IdlpProblemInfoModule,
    IdlpProblemAssetsModule,
    MatMenuModule,
    NgxsModule.forRoot([
      ProblemListState,
      ProblemInfoState,
      WsState,
      IdlpRootState,
    ]),
    NgxsWebsocketPluginModule.forRoot({
      url: environment.ws,
      typeKey: 'event',
      reconnectInterval: 5000,
      reconnectAttempts: 100,
    }),
    NgxsReduxDevtoolsPluginModule.forRoot({
      disabled: environment.production
    }),
    NgxsLoggerPluginModule.forRoot({
      disabled: environment.production
    }),
  ],
  providers: [],
  bootstrap: [
    IdlpRootComponent
  ]
})
export class IdlpRootModule {
}
