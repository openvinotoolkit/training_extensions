/**
 * @overview WebSocket module
 * @copyright (c) JSC Intel A/O
 */

import { ModuleWithProviders, NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IWSConfig, WebsocketService, WS_CONFIG } from '@idlp/providers/websocket.service';

@NgModule({
  imports: [
    CommonModule
  ],
  providers: [
    WebsocketService
  ]
})
export class WebsocketModule {
  public static forRoot(wsConfig: IWSConfig): ModuleWithProviders {
    return {
      ngModule: WebsocketModule,
      providers: [
        {
          provide: WS_CONFIG,
          useValue: wsConfig
        }
      ]
    };
  }
}
