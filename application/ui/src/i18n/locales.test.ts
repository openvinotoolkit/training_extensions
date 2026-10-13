// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import ts from 'typescript';

import { DEFAULT_LANGUAGE, resources, SUPPORTED_LANGUAGES } from './locales';
import en from './locales/en.json';

const sourceRoot = join(process.cwd(), 'src');
const sourceFiles = readdirSync(sourceRoot, { recursive: true, encoding: 'utf8' })
    .filter(
        (file) => /\.tsx?$/.test(file) && !/(\.test\.|\.spec\.|\.d\.ts$|test-utils|setup-tests|__mocks__)/.test(file)
    )
    .map((file) =>
        ts.createSourceFile(file, readFileSync(join(sourceRoot, file), 'utf8'), ts.ScriptTarget.Latest, true)
    );

const visitNodes = (node: ts.Node, visitor: (node: ts.Node) => void): void => {
    visitor(node);
    ts.forEachChild(node, (child) => visitNodes(child, visitor));
};

const flattenTranslations = (object: object, prefix = ''): [string, string][] =>
    Object.entries(object).flatMap(([key, value]) =>
        typeof value === 'string' ? [[`${prefix}${key}`, value]] : flattenTranslations(value, `${prefix}${key}.`)
    );

const translations = flattenTranslations(en);

const location = (source: ts.SourceFile, node: ts.Node): string =>
    `${source.fileName}:${source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1}`;

describe('locale registry', () => {
    it('registers a translation resource for every locale', () => {
        expect(Object.keys(resources)).toEqual(SUPPORTED_LANGUAGES);
        SUPPORTED_LANGUAGES.forEach((tag) => {
            expect(resources[tag]).toHaveProperty('translation');
        });
    });

    it('ships English as the default language', () => {
        expect(DEFAULT_LANGUAGE).toBe('en');
        expect(SUPPORTED_LANGUAGES).toEqual(['en', 'es', 'it', 'pl', 'pt', 'zh-CN', 'zh-HK', 'zh-MO', 'zh-TW']);
        expect(resources.en.translation).toHaveProperty('common.labels.dataset', 'Dataset');
    });
});

describe('translation coverage', () => {
    it('has no unused English keys, including plural variants and computed keys', () => {
        const references = new Set<string>();
        const dynamicPrefixes = new Set<string>();

        sourceFiles.forEach((source) => {
            visitNodes(source, (node) => {
                if (ts.isStringLiteralLike(node)) references.add(node.text);
                if (ts.isTemplateExpression(node) && node.head.text.includes('.')) {
                    dynamicPrefixes.add(node.head.text);
                }
            });
        });

        expect(
            translations
                .map(([key]) => key)
                .filter(
                    (key) =>
                        !references.has(key) &&
                        !references.has(key.replace(/_(zero|one|two|few|many|other|subset)$/, '')) &&
                        ![...dynamicPrefixes].some((prefix) => key.startsWith(prefix))
                )
        ).toEqual([]);
    });

    it('defines shared actions and labels only in common', () => {
        const commonValues = new Set(flattenTranslations(en.common).map(([, value]) => value));

        expect(translations.filter(([key, value]) => !key.startsWith('common.') && commonValues.has(value))).toEqual(
            []
        );
    });

    it('has no duplicated English values outside of deliberately context-specific keys', () => {
        // Keys whose English wording collides by accident but may diverge in other locales.
        const contextualDuplicates = new Set([
            'common.labels.training_subset',
            'common.labels.validation_subset',
            'models.performance.categories.accuracy',
            'jobs.tabs.finished',
            'jobs.status.pending',
            'common.upload.queued',
        ]);
        const keysByValue = new Map<string, string[]>();

        translations
            .filter(([key]) => !contextualDuplicates.has(key))
            .forEach(([key, value]) => keysByValue.set(value, [...(keysByValue.get(value) ?? []), key]));

        expect([...keysByValue].filter(([, keys]) => keys.length > 1)).toEqual([]);
    });

    it('does not silently overwrite duplicate JSON properties', () => {
        const source = ts.parseJsonText('en.json', readFileSync(join(sourceRoot, 'i18n/locales/en.json'), 'utf8'));
        const duplicates: string[] = [];

        visitNodes(source, (node) => {
            if (!ts.isObjectLiteralExpression(node)) return;

            const names = new Set<string>();
            node.properties.forEach((property) => {
                const name = property.name?.getText(source);
                if (name === undefined) return;
                if (names.has(name)) duplicates.push(`${location(source, property)} ${name}`);
                names.add(name);
            });
        });

        expect(duplicates).toEqual([]);
    });

    it('keeps explicit ARIA labels out of translation calls', () => {
        const translatedLabels: string[] = [];
        const containsTranslationCall = (source: ts.SourceFile, node: ts.Node): boolean => {
            let found = false;

            visitNodes(node, (child) => {
                if (ts.isCallExpression(child) && /^(t|i18n\.t)$/.test(child.expression.getText(source))) found = true;
            });

            return found;
        };

        sourceFiles.forEach((source) => {
            visitNodes(source, (node) => {
                if (ts.isJsxAttribute(node) && /aria.*label|cueLabel/i.test(node.name.getText(source))) {
                    if (containsTranslationCall(source, node)) translatedLabels.push(location(source, node));
                }

                // ARIA labels forwarded through object literals or variables, e.g. `{ 'aria-label': t(...) }`
                if (
                    ts.isPropertyAssignment(node) &&
                    /aria-?label/i.test(node.name.getText(source)) &&
                    containsTranslationCall(source, node.initializer)
                ) {
                    translatedLabels.push(location(source, node));
                }
            });
        });

        expect(translatedLabels).toEqual([]);
    });

    it('formats dates, numbers and lists with the active language instead of a hardcoded locale', () => {
        const hardcoded: string[] = [];

        sourceFiles.forEach((source) => {
            visitNodes(source, (node) => {
                if (!ts.isCallExpression(node) && !ts.isNewExpression(node)) return;
                if (
                    !/^(Intl\.(DateTimeFormat|NumberFormat|ListFormat|RelativeTimeFormat)|.*\.toLocale(Date|Time)?String)$/.test(
                        node.expression.getText(source)
                    )
                ) {
                    return;
                }

                const [localeArgument] = node.arguments ?? [];
                if (localeArgument && ts.isStringLiteralLike(localeArgument)) {
                    hardcoded.push(`${location(source, node)} ${localeArgument.text}`);
                }
            });
        });

        expect(hardcoded).toEqual([]);
    });

    it('translates visible JSX text and text props', () => {
        const technicalText = new Set(['Geti™', 'Geti v', ').zip', 'v', 'x', 'f', '&nbsp;']);
        const textProps =
            /^(label|title|placeholder|description|tooltip|errorMessage|alt|primaryActionLabel|secondaryActionLabel|cancelLabel|hotkey|message|bottomIconMessage|summary|emptyMessage)$/;
        const untranslated: string[] = [];

        sourceFiles.forEach((source) => {
            const inspectExpression = (expression: ts.Expression): string[] => {
                if (ts.isStringLiteralLike(expression)) return [expression.text];
                if (ts.isTemplateExpression(expression)) {
                    return [expression.head.text + expression.templateSpans.map((span) => span.literal.text).join('')];
                }
                if (ts.isConditionalExpression(expression)) {
                    return [...inspectExpression(expression.whenTrue), ...inspectExpression(expression.whenFalse)];
                }
                if (ts.isBinaryExpression(expression)) {
                    if (expression.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken) {
                        return inspectExpression(expression.right);
                    }
                    if (
                        [
                            ts.SyntaxKind.PlusToken,
                            ts.SyntaxKind.QuestionQuestionToken,
                            ts.SyntaxKind.BarBarToken,
                        ].includes(expression.operatorToken.kind)
                    ) {
                        return [...inspectExpression(expression.left), ...inspectExpression(expression.right)];
                    }
                }
                return [];
            };

            visitNodes(source, (node) => {
                let texts: string[] = [];

                if (ts.isJsxText(node)) texts = [node.text.trim()];
                if (ts.isJsxExpression(node) && node.expression) {
                    if (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent)) {
                        texts = inspectExpression(node.expression);
                    }
                }
                if (ts.isJsxAttribute(node) && textProps.test(node.name.getText(source))) {
                    const element = node.parent.parent;
                    if (
                        node.name.getText(source) === 'label' &&
                        (ts.isJsxOpeningElement(element) || ts.isJsxSelfClosingElement(element)) &&
                        ['ResizeAnchor', 'Anchor'].includes(element.tagName.getText(source))
                    ) {
                        return;
                    }
                    const isHidden = node.parent.properties.some(
                        (attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(source) === 'isHidden'
                    );
                    if (isHidden) return;

                    if (node.initializer && ts.isStringLiteral(node.initializer)) texts = [node.initializer.text];
                    if (node.initializer && ts.isJsxExpression(node.initializer)) {
                        const expression = node.initializer.expression;
                        if (expression) texts = inspectExpression(expression);
                    }
                }

                texts.forEach((text) => {
                    if (/[a-z]/i.test(text) && !technicalText.has(text)) {
                        untranslated.push(`${location(source, node)} ${text}`);
                    }
                });
            });
        });

        expect(untranslated).toEqual([]);
    });
});
