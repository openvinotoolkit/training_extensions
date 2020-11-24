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

const {SpecReporter} = require('jasmine-spec-reporter');

exports.config = {
	allScriptsTimeout: 11000,
	specs: [
		'./src/**/*.e2e-spec.ts'
	],
	capabilities: {
		browserName: 'chrome'
	},
	directConnect: true,
	framework: 'jasmine',
	jasmineNodeOpts: {
		showColors: true,
		defaultTimeoutInterval: 30000,
		// eslint-disable-next-line @typescript-eslint/explicit-function-return-type,prefer-arrow/prefer-arrow-functions
		print() {
		}
	},
	// eslint-disable-next-line @typescript-eslint/explicit-function-return-type,prefer-arrow/prefer-arrow-functions
	onPrepare() {
		require('ts-node').register({
			project: require('path').join(__dirname, './tsconfig.json')
		});
		jasmine.getEnv().addReporter(new SpecReporter({spec: {displayStacktrace: true}}));
	},
	params: {
		baseUrl: 'http://localhost:4200',
		cvat: {
			url: 'http://localhost:8080',
			credentials: {
				username: 'django',
				password: 'django'
			}
		}
	}
};
