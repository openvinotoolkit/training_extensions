/* eslint-disable header/header */
const applyInlineStyles = (sourceNode: Element, targetNode: Element) => {
    const computedStyle = window.getComputedStyle(sourceNode);

    // Loop through all computed styles and copy them
    for (const key of Array.from(computedStyle)) {
        const val = computedStyle.getPropertyValue(key);
        if (val) {
            const priority = computedStyle.getPropertyPriority(key);
            (targetNode as HTMLElement | SVGElement).style.setProperty(key, val, priority);
        }
    }

    // Some specific attributes that Recharts sets that need explicitly translating from CSS vars
    const presentationAttrs = ['stroke', 'fill'];
    presentationAttrs.forEach((attr) => {
        const attrValue = sourceNode.getAttribute(attr);
        if (attrValue && attrValue.includes('var(')) {
            const match = attrValue.match(/var\(([^)]+)\)/);
            if (match) {
                const varName = match[1];

                // We need to resolve the CSS variable.
                // Checking a parent if the property isn't defined directly on the SVG element.
                let resolvedValue = computedStyle.getPropertyValue(varName);
                if (!resolvedValue && sourceNode.parentElement) {
                    // Find the closest ancestor that has the property defined.
                    // In this case, Recharts may use CSS vars defined in a higher scope.
                    let parent: Element | null = sourceNode;
                    while (parent && !resolvedValue) {
                        resolvedValue = window.getComputedStyle(parent).getPropertyValue(varName);
                        parent = parent.parentElement;
                    }
                }

                if (resolvedValue) {
                    targetNode.setAttribute(attr, resolvedValue);
                    (targetNode as HTMLElement | SVGElement).style.setProperty(attr, resolvedValue, 'important');
                }
            }
        }
    });
};

export const downloadSvgAsImage = (svgElement: SVGSVGElement, title: string): Promise<void> => {
    return new Promise((resolve, reject) => {
        try {
            // Clone the SVG so we don't modify the original
            const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

            // Ensure inline styles are applied correctly by setting explicit width/height
            const bbox = svgElement.getBBox();
            const width = bbox.width + Math.abs(bbox.x);
            const height = bbox.height + Math.abs(bbox.y);

            clonedSvg.setAttribute('width', `${width}`);
            clonedSvg.setAttribute('height', `${height}`);
            clonedSvg.setAttribute('viewBox', `${bbox.x} ${bbox.y} ${width} ${height}`);

            const sourceElements = Array.from(svgElement.querySelectorAll('*'));
            const targetElements = Array.from(clonedSvg.querySelectorAll('*'));

            sourceElements.forEach((sourceEl, index) => {
                applyInlineStyles(sourceEl, targetElements[index]);
            });

            applyInlineStyles(svgElement, clonedSvg);

            const svgString = new XMLSerializer().serializeToString(clonedSvg);
            const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(blob);

            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;

                const ctx = canvas.getContext('2d');
                if (ctx) {
                    // White background
                    ctx.fillStyle = '#ffffff';
                    ctx.fillRect(0, 0, width, height);
                    ctx.drawImage(img, 0, 0);

                    const dataUrl = canvas.toDataURL('image/png');
                    const link = document.createElement('a');
                    link.href = dataUrl;
                    link.download = `${title.replace(/\s+/g, '_').toLowerCase()}_metrics.png`;
                    link.click();
                }

                URL.revokeObjectURL(url);
                resolve();
            };
            img.onerror = (err) => {
                URL.revokeObjectURL(url);
                reject(err);
            };
            img.src = url;
        } catch (error) {
            reject(error);
        }
    });
};
