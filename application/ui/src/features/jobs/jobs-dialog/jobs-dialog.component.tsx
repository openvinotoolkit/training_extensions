// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { TrainingLogsDialog } from '@/components/training-logs/training-logs-dialog.component';
import { useTranslation } from '@/i18n';
import { ActionButton, Content, Dialog, DialogContainer, Divider, Flex, Header, Heading } from '@geti-ui/ui';
import { CloseSemiBold } from '@geti-ui/ui/icons';
import { useModelJobs } from 'hooks/api/jobs/jobs.hook';
import { useProjects } from 'hooks/api/project.hook';

import { useArchitectureNames } from '../hooks/use-architecture-names.hook';
import {
    ALL_PROJECTS,
    countByStatusGroup,
    filterJobs,
    getStatusGroup,
    sortJobs,
    type ModelJob,
    type StatusGroup,
} from '../utils';
import { JobStatusTabs } from './job-status-tabs.component';
import { JobsList } from './jobs-list.component';
import { ProjectFilterPicker } from './project-filter-picker.component';

type JobsDialogProps = {
    initialProjectId: string;
    currentProjectId?: string;
    onClose: () => void;
};

export const JobsDialog = ({ initialProjectId, currentProjectId, onClose }: JobsDialogProps) => {
    const { t } = useTranslation();
    const { data: projects } = useProjects();
    const { data: jobs = [] } = useModelJobs();
    const architectureName = useArchitectureNames(projects);

    const [projectId, setProjectId] = useState(initialProjectId);
    const [statusGroup, setStatusGroup] = useState<StatusGroup>('all');
    const [logsJob, setLogsJob] = useState<ModelJob | null>(null);

    const projectName = (id: string) => projects.find((project) => project.id === id)?.name ?? id;

    const scoped = filterJobs(jobs, projectId, 'all');
    const counts = countByStatusGroup(scoped);
    const visible = filterJobs(sortJobs(scoped), ALL_PROJECTS, statusGroup);
    const isAllProjects = projectId === ALL_PROJECTS;

    return (
        <DialogContainer onDismiss={onClose}>
            <Dialog width={'90vw'} height={'80vh'}>
                <Heading>{t('common.labels.jobs')}</Heading>
                <Header>
                    {projects.length > 1 && (
                        <ProjectFilterPicker
                            projects={projects}
                            value={projectId}
                            onChange={setProjectId}
                            currentProjectId={currentProjectId}
                        />
                    )}
                    <ActionButton isQuiet aria-label={'Close jobs'} marginStart={'size-200'} onPress={onClose}>
                        <CloseSemiBold />
                    </ActionButton>
                </Header>
                <Divider />
                <Content>
                    <Flex direction={'column'} height={'100%'} gap={'size-200'}>
                        <JobStatusTabs counts={counts} selected={statusGroup} onChange={setStatusGroup} />
                        <JobsList
                            jobs={visible}
                            isAllProjects={isAllProjects}
                            showStatusBadge={statusGroup === 'all'}
                            statusGroup={statusGroup}
                            projectName={projectName}
                            architectureName={architectureName}
                            onViewLogs={setLogsJob}
                        />
                    </Flex>

                    <DialogContainer type={'fullscreen'} onDismiss={() => setLogsJob(null)}>
                        {logsJob && (
                            <TrainingLogsDialog
                                projectId={logsJob.metadata.project.id}
                                jobId={logsJob.job_id}
                                isJobActive={getStatusGroup(logsJob.status) === 'running'}
                            />
                        )}
                    </DialogContainer>
                </Content>
            </Dialog>
        </DialogContainer>
    );
};
