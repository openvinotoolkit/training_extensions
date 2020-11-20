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
import {IProblem} from '@idlp/routed/problem-list/problem-list.models';
import {Action, Selector, State, StateContext} from '@ngxs/store';
import {ImmutableSelector} from '@ngxs-labs/immer-adapter';
import {ProblemCreate, ProblemDelete, ProblemList} from './problem-list.actions';

export class ProblemListStateModel {
  public items: IProblem[];
}

const defaults = {
  items: [],
};

@State<ProblemListStateModel>({
  name: 'problemList',
  defaults
})
@Injectable()
export class ProblemListState {
  @Selector()
  @ImmutableSelector()
  static getProblemList(state: ProblemListStateModel): IProblem[] {
    return state.items;
  }

  @Action(ProblemList)
  list({setState}: StateContext<ProblemListStateModel>, {data}: ProblemList): void {
    const problems = data.items.filter((item: IProblem) => item.type !== 'generic');
    data.items.map((item: IProblem) => {
      if (item.type === 'generic') problems.push(item);
    });
    setState({items: problems});
  }

  @Action(ProblemCreate)
  add({getState, setState}: StateContext<ProblemListStateModel>, {data}: ProblemCreate): void {
    const state = getState();
    const problems = state.items.filter((item: IProblem) => item.type !== 'generic');
    problems.push(data);
    state.items.map((item: IProblem) => {
      if (item.type === 'generic') problems.push(item);
    });
    setState({items: problems});
  }

  @Action(ProblemDelete)
  delete({getState, setState}: StateContext<ProblemListStateModel>, {data}: ProblemDelete): void {
    const state = getState();
    const problems = state.items.filter((item: IProblem) => item.id !== data.id);
    setState({items: problems});
  }
}
