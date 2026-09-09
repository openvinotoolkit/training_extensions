// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { vi } from 'vitest';

import { downloadSvgAsImage } from './download-graph.utils';

describe('downloadSvgAsImage', () => {
    let mockSvg: SVGSVGElement;
    let createObjectURLSpy: ReturnType<typeof vi.fn>;
    let revokeObjectURLSpy: ReturnType<typeof vi.fn>;
    let anchorClickSpy: ReturnType<typeof vi.fn>;
    let mockContext2D: CanvasRenderingContext2D;

    beforeEach(() => {
        // Mock SVG Element and its getBBox
        mockSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        mockSvg.getBBox = vi.fn().mockReturnValue({ x: 0, y: 0, width: 100, height: 100 });

        // Add a child to test applyInlineStyles recursion
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('fill', 'var(--test-color)');
        mockSvg.appendChild(rect);

        // Mock getComputedStyle
        vi.spyOn(window, 'getComputedStyle').mockImplementation((_node) => {
            return {
                getPropertyValue: (prop: string) => {
                    if (prop === '--test-color') return '#123456';
                    return '';
                },
                getPropertyPriority: () => '',
            } as unknown as CSSStyleDeclaration;
        });

        // Mock URL methods
        createObjectURLSpy = vi.fn().mockReturnValue('blob:test-url');
        global.URL.createObjectURL = createObjectURLSpy as unknown as typeof URL.createObjectURL;
        revokeObjectURLSpy = vi.fn();
        global.URL.revokeObjectURL = revokeObjectURLSpy as unknown as typeof URL.revokeObjectURL;

        // Mock canvas context
        mockContext2D = {
            fillStyle: '',
            fillRect: vi.fn(),
            drawImage: vi.fn(),
        } as unknown as CanvasRenderingContext2D;

        // Mock document.createElement to handle canvas and anchor
        anchorClickSpy = vi.fn();
        const originalCreateElement = document.createElement.bind(document);
        vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
            if (tagName === 'canvas') {
                return {
                    width: 0,
                    height: 0,
                    getContext: vi.fn().mockReturnValue(mockContext2D),
                    toDataURL: vi.fn().mockReturnValue('data:image/png;base64,test'),
                } as unknown as HTMLCanvasElement;
            }
            if (tagName === 'a') {
                return {
                    href: '',
                    download: '',
                    click: anchorClickSpy,
                } as unknown as HTMLAnchorElement;
            }
            // Fallback for other elements
            return originalCreateElement(tagName) as HTMLElement;
        });

        // Override Image to trigger onload synchronously
        const OriginalImage = global.Image;
        global.Image = class extends OriginalImage {
            constructor() {
                super();
                setTimeout(() => {
                    if (this.onload) this.onload(new Event('load'));
                }, 0);
            }
        } as unknown as typeof Image;
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should successfully clone SVG, draw to canvas, and trigger download', async () => {
        const title = 'Test Graph title';
        await downloadSvgAsImage(mockSvg, title);

        expect(createObjectURLSpy).toHaveBeenCalled();
        expect(mockContext2D.fillRect).toHaveBeenCalledWith(0, 0, 100, 100);
        expect(mockContext2D.drawImage).toHaveBeenCalled();

        // Assert download action
        expect(anchorClickSpy).toHaveBeenCalled();

        // The URL should be revoked after the image is loaded
        expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:test-url');
    });

    it('should handle image loading errors gracefully', async () => {
        // Override Image to trigger onerror
        global.Image = class {
            onerror: ((err: Error) => void) | null = null;
            onload: (() => void) | null = null;
            src: string = '';
            constructor() {
                setTimeout(() => {
                    if (this.onerror) this.onerror(new Error('Image failed to load'));
                }, 0);
            }
        } as unknown as typeof Image;

        const title = 'Test Error Graph';

        await expect(downloadSvgAsImage(mockSvg, title)).rejects.toThrow('Image failed to load');

        // Ensure URL is revoked even on error
        expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:test-url');
    });
});
