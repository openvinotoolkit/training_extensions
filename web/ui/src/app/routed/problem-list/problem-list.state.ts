/**
 * @overview
 * @copyright (c) JSC Intel A/O
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
  add({getState, setState}: StateContext<ProblemListStateModel>, {problem}: ProblemCreate): void {
    const state = getState();
    const problems = state.items.filter((item: IProblem) => item.type !== 'generic');
    problems.push(problem);
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
