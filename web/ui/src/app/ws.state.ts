import {Injectable} from '@angular/core';
import {Action, Selector, State, StateContext} from '@ngxs/store';
import {ConnectWebSocket, WebSocketConnected, WebSocketDisconnected} from '@ngxs/websocket-plugin';
import {ImmutableSelector} from '@ngxs-labs/immer-adapter';
import {WsConnect} from './ws.actions';


export class WsStateModel {
  public connected: boolean;
}

const defaults = {
  connected: false
};

@State<WsStateModel>({
  name: 'ws',
  defaults
})
@Injectable()
export class WsState {

  @Selector()
  @ImmutableSelector()
  static connected(state: WsStateModel): boolean {
    return state.connected;
  }

  @Action(WsConnect)
  wsConnect({getState, dispatch}: StateContext<WsStateModel>, {force}: WsConnect) {
    const state = getState();
    if (state.connected && !force) {
      return;
    }
    dispatch(new ConnectWebSocket());
  }

  @Action(WebSocketConnected)
  webSocketConnected({patchState}: StateContext<WsStateModel>) {
    patchState({connected: true});
  }

  @Action(WebSocketDisconnected)
  webSocketDisconnected({patchState}: StateContext<WsStateModel>) {
    patchState({connected: false});
  }
}
