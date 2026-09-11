// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useTranslation, type TranslateFn } from '@/i18n';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Cell,
    Column,
    Content,
    Dialog,
    DialogContainer,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Loading,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { AcceptCircle, CrossCircle, Pending } from '@geti-ui/ui/icons';

import { formatBytes } from '../../../../shared/util';
import { useMediaUploadDispatch, useMediaUploadState } from '../../providers/media-upload-provider.component';
import { computeSummary, type UploadFileItem, type UploadItemStatus } from '../../providers/media-upload-reducer';

import classes from './upload-details-dialog.module.scss';

const StatusIcon = ({ status }: { status: UploadItemStatus }): ReactNode => {
    switch (status) {
        case 'queued':
            return <Pending aria-label={'Queued'} size={'S'} />;
        case 'uploading':
            return <Loading mode={'inline'} size={'S'} />;
        case 'uploaded':
            return (
                <AcceptCircle aria-label={'Uploaded'} width={16} height={16} style={{ fill: 'var(--brand-moss)' }} />
            );
        case 'failed':
            return (
                <CrossCircle
                    aria-label={'Failed'}
                    width={16}
                    height={16}
                    style={{ fill: 'var(--brand-coral-cobalt)' }}
                />
            );
    }
};

const StatusCell = ({
    item,
    labels,
    t,
}: {
    item: UploadFileItem;
    labels: Record<UploadItemStatus, string>;
    t: TranslateFn;
}) => {
    const statusContent = (
        <Flex alignItems={'center'} gap={'size-100'}>
            <StatusIcon status={item.status} />
            <Text>{labels[item.status]}</Text>
        </Flex>
    );

    if (item.status === 'failed' && item.errorMessage) {
        return (
            <Flex alignItems={'center'} gap={'size-100'}>
                {statusContent}
                <DialogTrigger type={'popover'}>
                    <ActionButton isQuiet aria-label={'Error details'} UNSAFE_className={classes.error}>
                        {t('dataset.upload.error')}
                    </ActionButton>
                    <Dialog>
                        <Heading>{t('dataset.upload.errorTitle')}</Heading>
                        <Divider />
                        <Content>
                            <Text>{item.errorMessage}</Text>
                        </Content>
                    </Dialog>
                </DialogTrigger>
            </Flex>
        );
    }

    return statusContent;
};

const buildSubheader = (
    t: TranslateFn,
    total: number,
    succeeded: number,
    failed: number,
    isUploading: boolean
): string => {
    if (isUploading) {
        return t(failed > 0 ? 'dataset.upload.uploadingFailedSummary' : 'dataset.upload.uploadingSummary', {
            total,
            uploaded: succeeded,
            failed,
        });
    }

    if (failed === 0) return t('dataset.upload.uploadedSummary', { count: succeeded });
    if (succeeded === 0) return t('dataset.upload.failedSummary', { count: failed });

    return t('dataset.upload.mixedSummary', { uploaded: succeeded, failed });
};

const UploadDetailsDialogContent = ({ onClose }: { onClose: () => void }) => {
    const { t } = useTranslation();
    const labels: Record<UploadItemStatus, string> = {
        queued: t('dataset.upload.queued'),
        uploading: t('dataset.upload.uploading'),
        uploaded: t('dataset.upload.uploaded'),
        failed: t('dataset.upload.failed'),
    };
    const state = useMediaUploadState();
    const summary = computeSummary(state.items);
    const items = state.items;

    const subheader = buildSubheader(t, summary.total, summary.succeeded, summary.failed, state.isUploading);

    return (
        <Dialog size={'L'}>
            <Heading>{t('dataset.upload.details')}</Heading>
            <Divider />
            <Content>
                <Flex direction={'column'} gap={'size-200'}>
                    <Text>{subheader}</Text>
                    <TableView
                        aria-label={'Upload details'}
                        overflowMode={'truncate'}
                        density={'compact'}
                        maxHeight={'60vh'}
                        isQuiet
                    >
                        <TableHeader>
                            <Column isRowHeader>{t('dataset.upload.filename')}</Column>
                            <Column width={160}>{t('dataset.upload.status')}</Column>
                            <Column width={120} align={'end'}>
                                {t('dataset.upload.size')}
                            </Column>
                        </TableHeader>
                        <TableBody items={items}>
                            {(item) => (
                                <Row key={item.id}>
                                    <Cell>
                                        <TooltipTrigger>
                                            <Text>{item.name}</Text>
                                            <Tooltip>{item.name}</Tooltip>
                                        </TooltipTrigger>
                                    </Cell>
                                    <Cell>
                                        <StatusCell item={item} labels={labels} t={t} />
                                    </Cell>
                                    <Cell>{formatBytes(item.size)}</Cell>
                                </Row>
                            )}
                        </TableBody>
                    </TableView>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant={'primary'} onPress={onClose}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const UploadDetailsDialog = () => {
    const state = useMediaUploadState();
    const dispatch = useMediaUploadDispatch();
    const close = () => dispatch({ type: 'CLOSE_DIALOG' });

    return (
        <DialogContainer onDismiss={close}>
            {state.isDetailsDialogOpen && <UploadDetailsDialogContent onClose={close} />}
        </DialogContainer>
    );
};
