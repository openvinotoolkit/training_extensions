/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Injectable} from '@angular/core';
import {environment} from '@environments/environment';
import {Action, Selector, State, StateContext} from '@ngxs/store';
import {SetTheme} from './idlp-root.actions';

export class IdlpRootStateModel {
  public theme: string;
}

const getTheme = (): string => {
  const selectedTheme: string = localStorage.getItem('theme');
  if (selectedTheme?.length && (selectedTheme === environment.themes.light || selectedTheme === environment.themes.dark)) {
    return selectedTheme;
  } else {
    return environment.themeDefault;
  }
};

const defaults = {
  theme: getTheme(),
};


@State<IdlpRootStateModel>({
  name: 'idlpRoot',
  defaults
})
@Injectable()
export class IdlpRootState {
  @Selector()
  static theme(state: IdlpRootStateModel): string {
    return state.theme;
  }

  @Action(SetTheme)
  add({getState, setState}: StateContext<IdlpRootStateModel>, {theme}: SetTheme): void {
    const state = getState();
    localStorage.setItem('theme', theme);
    setState({
      ...state,
      theme
    });
  }
}
