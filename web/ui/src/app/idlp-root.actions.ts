export class SetTheme {
  static readonly type = '[Theme] Set';

  constructor(public theme: string) {
  }
}
