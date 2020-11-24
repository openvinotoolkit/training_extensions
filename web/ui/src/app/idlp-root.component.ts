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

import {OverlayContainer} from '@angular/cdk/overlay';
import {Component, HostBinding, OnDestroy, OnInit} from '@angular/core';
import {NavigationEnd, Router, RouterEvent} from '@angular/router';
import {environment} from '@environments/environment';
import {SetTheme} from '@idlp/root/idlp-root.actions';
import {IdlpRootState} from '@idlp/root/idlp-root.state';
import {Select, Store} from '@ngxs/store';
import {Observable, Subject} from 'rxjs';
import {filter, takeUntil} from 'rxjs/operators';

interface IMenuItem {
  title: string;
  path: string;
}

@Component({
  selector: 'idlp-root',
  templateUrl: './idlp-root.component.html',
  styleUrls: ['./idlp-root.component.scss']
})
export class IdlpRootComponent implements OnInit, OnDestroy {
  @Select(IdlpRootState.theme) themeActive$: Observable<string>;

  @HostBinding('class')
  componentCssClass: string;

  problemId: string;
  problemContext: boolean;
  selectedPath: string;

  navMenu: IMenuItem[] = [
    {title: 'Info', path: 'info'},
    {title: 'Assets', path: 'assets'}
  ];

  title: string = environment.title;
  private destroy$: Subject<any> = new Subject();

  constructor(
    private router: Router,
    private overlayContainer: OverlayContainer,
    private store: Store,
  ) {
    // @ts-ignore
    console.log(`version ${process.env.IDLP_VERSION}`);
  }

  ngOnInit(): void {
    this.themeActive$
      .pipe(takeUntil(this.destroy$))
      .subscribe((theme: string) => {
          document.body.classList.remove(this.componentCssClass);
          document.body.classList.add(theme);
          this.overlayContainer.getContainerElement().classList.remove(this.componentCssClass);
          this.overlayContainer.getContainerElement().classList.add(theme);
          this.componentCssClass = theme;
        }
      );

    this.router.events
      .pipe(
        takeUntil(this.destroy$),
        filter((event: RouterEvent) => event instanceof NavigationEnd)
      )
      .subscribe(() => {
        this.problemId = this.router.url.split('/')[1];
        this.problemContext = this.problemId?.length > 0;
        this.selectedPath = this.router.url.split('/')[2];
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  setTheme(theme) {
    switch (theme) {
    case 'light':
      this.store.dispatch(new SetTheme(environment.themes.light));
      break;
    case 'dark':
      this.store.dispatch(new SetTheme(environment.themes.dark));
      break;
    default:
      return;
    }
  }
}
