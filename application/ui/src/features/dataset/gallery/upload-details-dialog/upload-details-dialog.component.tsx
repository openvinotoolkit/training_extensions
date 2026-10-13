// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { getUploadPercentage } from '@/components/upload-progress/upload-progress.component';
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
    ProgressBar,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { AcceptCircle, CloseSmall, CrossCircle, Pending } from '@geti-ui/ui/icons';

import { formatBytes } from '../../../../shared/util';
import { useUploadActions } from '../../hooks/use-upload-actions';
import { useMediaUploadDispatch, useMediaUploadState } from '../../providers/media-upload-context';
import {
    computeSummary,
    formatFinalSummary,
    isCancellable,
    type UploadFileItem,
    type UploadItemStatus,
} from '../../providers/media-upload-reducer';

import classes from './upload-details-dialog.module.scss';

const StatusIcon = ({ status }: { status: UploadItemStatus }): ReactNode => {
    switch (status) {
        case 'queued':
            return <Pending aria-label={'Queued'} size={'S'} />;
        case 'uploading':
        case 'processing':
            return <Loading mode={'inline'} size={'S'} />;
        case 'uploaded':
            return (
                <AcceptCircle aria-label={'Uploaded'} width={16} height={16} style={{ fill: 'var(--brand-moss)' }} />
            );
        case 'cancelled':
            return <CloseSmall aria-label={'Cancelled'} width={16} height={16} />;
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
    if (item.status === 'uploading' && item.bytesSent !== undefined) {
        return (
            <Flex alignItems={'center'} gap={'size-100'}>
                <ProgressBar
                    aria-label={'Upload progress'}
                    size={'S'}
                    width={'size-1200'}
                    value={getUploadPercentage(item.bytesSent, item.size)}
                />
                <Text>
                    {t('common.labels.bytesTransferred', {
                        transferred: formatBytes(item.bytesSent),
                        total: formatBytes(item.size),
                    })}
                </Text>
            </Flex>
        );
    }

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
                        {t('common.status.error')}
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

const CancelCell = ({ item, onCancel }: { item: UploadFileItem; onCancel: (itemId: string) => void }) => {
    if (!isCancellable(item.status)) {
        return null;
    }

    return (
        <ActionButton isQuiet aria-label={`Cancel upload of ${item.name}`} onPress={() => onCancel(item.id)}>
            <CloseSmall />
        </ActionButton>
    );
};

const buildSubheader = (
    t: TranslateFn,
    total: number,
    succeeded: number,
    failed: number,
    cancelled: number,
    isUploading: boolean
): string => {
    if (isUploading) {
        return t(failed > 0 ? 'dataset.upload.uploadingFailedSummary' : 'dataset.upload.uploadingSummary', {
            count: total,
            total,
            uploaded: succeeded,
            failed,
        });
    }

    return formatFinalSummary(t, { succeeded, failed, cancelled });
};

const UploadDetailsDialogContent = ({ onClose }: { onClose: () => void }) => {
    const { t } = useTranslation();
    const labels: Record<UploadItemStatus, string> = {
        queued: t('dataset.upload.queued'),
        uploading: t('common.status.uploading'),
        processing: t('dataset.upload.processing'),
        uploaded: t('dataset.upload.uploaded'),
        failed: t('common.status.failed'),
        cancelled: t('common.status.cancelled'),
    };
    const state = useMediaUploadState();
    const { cancelItems } = useUploadActions();
    const summary = computeSummary(state.items);
    const items = state.items;

    const subheader = buildSubheader(
        t,
        summary.total,
        summary.succeeded,
        summary.failed,
        summary.cancelled,
        state.isUploading
    );
    const cancellableItemIds = items.filter((item) => isCancellable(item.status)).map((item) => item.id);

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
                            <Column width={260}>{t('common.labels.statusUppercase')}</Column>
                            <Column width={100} align={'end'}>
                                {t('common.labels.sizeUppercase')}
                            </Column>
                            <Column width={48} align={'end'} hideHeader>
                                {t('common.actions.cancel')}
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
                                    <Cell>
                                        <CancelCell item={item} onCancel={(itemId) => cancelItems([itemId])} />
                                    </Cell>
                                </Row>
                            )}
                        </TableBody>
                    </TableView>
                </Flex>
            </Content>
            <ButtonGroup>
                {cancellableItemIds.length > 0 && (
                    <Button variant={'secondary'} onPress={() => cancelItems(cancellableItemIds)}>
                        {t('dataset.upload.cancelAll')}
                    </Button>
                )}
                <Button variant={'primary'} onPress={onClose}>
                    {t('common.actions.close')}
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
