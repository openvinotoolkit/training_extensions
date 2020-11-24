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
import {Inject, Injectable, InjectionToken, OnDestroy} from '@angular/core';
import {interval, Observable, Observer, Subject, Subscriber, SubscriptionLike} from 'rxjs';
import {distinctUntilChanged, filter, map, share, takeWhile} from 'rxjs/operators';
import {WebSocketSubject, WebSocketSubjectConfig} from 'rxjs/webSocket';

export const WS_CONFIG: InjectionToken<string> = new InjectionToken('websocket');

export interface IWSMessage<T> {
  event: string;
  data?: T;
}

export interface IWSConfig {
  url: string;
  reconnectInterval?: number;
  reconnectAttempts?: number;
}

export interface IWSService {
  on<T>(event: string): Observable<T>;

  send(event: string, data: any): void;
}

@Injectable({
  providedIn: 'root'
})
export class WebsocketService implements IWSService, OnDestroy {
  private config: WebSocketSubjectConfig<IWSMessage<any>>;

  private websocketSub: SubscriptionLike;
  private statusSub: SubscriptionLike;

  private status$: Observable<boolean>;
  private connection$: Observer<boolean>;
  private reconnection$: Observable<number>;

  private websocket$: WebSocketSubject<IWSMessage<any>>;
  private wsMessages$: Subject<IWSMessage<any>>;

  private reconnectInterval: number;
  private reconnectAttempts: number;
  private isConnected: boolean;

  private deferredMessagesQueue: IWSMessage<any>[] = [];

  constructor(@Inject(WS_CONFIG) private webSocketConfig: IWSConfig) {
    this.wsMessages$ = new Subject<IWSMessage<any>>();

    this.reconnectInterval = webSocketConfig.reconnectInterval || 5000;
    this.reconnectAttempts = webSocketConfig.reconnectAttempts || 10;

    this.config = {
      url: webSocketConfig.url,
      closeObserver: {
        next: (): void => {
          this.websocket$ = null;
          this.connection$.next(false);
          // eslint-disable-next-line no-console
          console.log('WebSocket disconnected');
        }
      },
      openObserver: {
        next: (): void => {
          this.connection$.next(true);
          // eslint-disable-next-line no-console
          console.log('WebSocket connected');
        }
      }
    };

    this.status$ = new Observable<boolean>((observer: Subscriber<boolean>) => {
      this.connection$ = observer;
    })
      .pipe(
        share(),
        distinctUntilChanged()
      );

    this.statusSub = this.status$
      .subscribe((isConnected: boolean) => {
        this.isConnected = isConnected;

        if (!this.reconnection$ && typeof (isConnected) === 'boolean' && !isConnected) {
          this.reconnect();
          return;
        }

        while (this.deferredMessagesQueue.length) {
          const deferredMessage = this.deferredMessagesQueue.shift();
          this.websocket$.next({event: deferredMessage.event, data: deferredMessage.data});
        }
      });

    this.websocketSub = this.wsMessages$
      .subscribe(null, (error: ErrorEvent) => {
        // eslint-disable-next-line no-console
        console.error('WebSocket error', error);
      });

    this.connect();
  }

  ngOnDestroy(): void {
    this.websocketSub.unsubscribe();
    this.statusSub.unsubscribe();
  }

  on<T>(event: string): Observable<T> {
    if (event) {
      return this.wsMessages$
        .pipe(
          filter((message: IWSMessage<T>) => message.event === event),
          map((message: IWSMessage<T>) => message.data)
        );
    }
  }

  send(event: string, data: any = {}): void {
    if (event) {
      if (this.isConnected) {
        this.websocket$.next({event, data});
      } else {
        this.deferredMessagesQueue.push({event, data});
      }
    } else {
      // eslint-disable-next-line no-console
      console.error('Message sending error');
    }
  }

  private connect(): void {
    this.websocket$ = new WebSocketSubject(this.config);

    this.websocket$
      .subscribe(
        (message: IWSMessage<any>) => this.wsMessages$.next(message),
        () => !this.websocket$ && this.reconnect()
      );
  }

  private reconnect(): void {
    // eslint-disable-next-line no-console
    console.log('WebSocket reconnecting');

    this.reconnection$ = interval(this.reconnectInterval)
      .pipe(takeWhile((v: number, index: number) => index < this.reconnectAttempts && !this.websocket$));

    this.reconnection$
      .subscribe(
        () => this.connect(),
        null,
        () => {
          this.reconnection$ = null;
          if (!this.websocket$) {
            this.wsMessages$.complete();
            this.connection$.complete();
          }
        });
  }
}
