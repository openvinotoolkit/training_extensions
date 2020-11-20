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
