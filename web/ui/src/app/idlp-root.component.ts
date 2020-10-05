/**
 * @overview
 * @copyright (c) JSC Intel A/O
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
