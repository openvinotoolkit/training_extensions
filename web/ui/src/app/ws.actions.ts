export class WsConnect {
  static readonly type = '[Ws] Connect';

  constructor(public force: boolean = false) {
  }
}
