// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const BUTTON_LEFT = {
    button: 0,
    buttons: 1,
};

const BUTTON_RIGHT = {
    button: 2,
    buttons: 2,
};

const BUTTON_WHEEL = {
    button: 1,
    buttons: 4,
};

const BUTTON_ERASER = {
    button: 5,
    buttons: 32,
};

export type MouseButton = {
    button: number;
    buttons: number;
};

const isButton = (button: MouseButton, buttonToCompare: MouseButton): boolean =>
    button.button === buttonToCompare.button || button.buttons === buttonToCompare.buttons;

export const isLeftButton = (button: MouseButton): boolean => {
    return isButton(button, BUTTON_LEFT);
};

export const isWheelButton = (button: MouseButton): boolean => {
    return isButton(button, BUTTON_WHEEL);
};

export const isEraserOrRightButton = (button: MouseButton): boolean => {
    return isButton(button, BUTTON_ERASER) || isButton(button, BUTTON_RIGHT);
};
