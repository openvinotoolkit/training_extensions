// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import { isFunction } from 'lodash-es';
import { render } from 'test-utils/render';

import { FrameStep } from './frame-step.component';

describe('FrameStep', () => {
    it('toggles to all frames', async () => {
        const setStepState = rs.fn();
        const currentStep = 60;
        const setStep: Dispatch<SetStateAction<number>> = (updateStep) => {
            setStepState(isFunction(updateStep) ? updateStep(currentStep) : updateStep);
        };
        render(<FrameStep isDisabled={false} step={currentStep} onChangeStep={setStep} defaultFps={currentStep} />);

        expect(screen.getByTestId('frame-mode-indicator-id')).toHaveTextContent('1/1');
        fireEvent.click(screen.getByRole('button'));

        expect(setStepState).toHaveBeenCalledWith(1);
    });

    it('toggles to 1 frame per second', async () => {
        const setStepState = rs.fn();
        const step = 1;
        const setStep: Dispatch<SetStateAction<number>> = (updateStep) => {
            setStepState(isFunction(updateStep) ? updateStep(step) : updateStep);
        };

        render(<FrameStep isDisabled={false} step={step} onChangeStep={setStep} defaultFps={60} />);

        expect(screen.getByTestId('frame-mode-indicator-id')).toHaveTextContent('ALL');
        fireEvent.click(screen.getByRole('button'));

        expect(setStepState).toHaveBeenCalledWith(60);
    });
});
