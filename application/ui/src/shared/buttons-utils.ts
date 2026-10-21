// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const BUTTON_LEFT = {
    button: 0,
    buttons: 1,
};

const BUTTON_WHEEL = {
    button: 1,
    buttons: 4,
};

type MouseButton = {
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
