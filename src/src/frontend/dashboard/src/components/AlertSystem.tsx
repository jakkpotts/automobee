import { Fragment, useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { XMarkIcon, BellIcon } from '@heroicons/react/24/outline';

interface Alert {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'warning' | 'error';
  timestamp: string;
  isRead: boolean;
}

const MOCK_ALERTS: Alert[] = [
  {
    id: '1',
    title: 'Camera Offline',
    message: 'Camera #123 at Main Street has gone offline.',
    type: 'error',
    timestamp: new Date().toISOString(),
    isRead: false,
  },
  {
    id: '2',
    title: 'High Detection Rate',
    message: 'Unusual traffic detected at Downtown intersection.',
    type: 'warning',
    timestamp: new Date().toISOString(),
    isRead: false,
  },
  {
    id: '3',
    title: 'System Update',
    message: 'New detection model version available.',
    type: 'info',
    timestamp: new Date().toISOString(),
    isRead: true,
  },
];

export function AlertSystem() {
  const [isOpen, setIsOpen] = useState(false);
  const [alerts] = useState<Alert[]>(MOCK_ALERTS);

  const unreadCount = alerts.filter(alert => !alert.isRead).length;

  return (
    <>
      {/* Alert Toggle Button */}
      <button
        type="button"
        className="fixed bottom-4 right-4 inline-flex items-center p-3 rounded-full shadow-lg bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        onClick={() => setIsOpen(true)}
      >
        <BellIcon className="h-6 w-6 text-gray-600" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 bg-red-600 rounded-full">
            {unreadCount}
          </span>
        )}
      </button>

      <Transition.Root show={isOpen} as={Fragment}>
        <Dialog as="div" className="relative z-50" onClose={setIsOpen}>
          <Transition.Child
            as={Fragment}
            enter="ease-in-out duration-500"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in-out duration-500"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-hidden">
            <div className="absolute inset-0 overflow-hidden">
              <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
                <Transition.Child
                  as={Fragment}
                  enter="transform transition ease-in-out duration-500"
                  enterFrom="translate-x-full"
                  enterTo="translate-x-0"
                  leave="transform transition ease-in-out duration-500"
                  leaveFrom="translate-x-0"
                  leaveTo="translate-x-full"
                >
                  <Dialog.Panel className="pointer-events-auto w-screen max-w-md">
                    <div className="flex h-full flex-col overflow-y-scroll bg-white shadow-xl">
                      <div className="px-4 py-6 sm:px-6">
                        <div className="flex items-start justify-between">
                          <Dialog.Title className="text-lg font-semibold text-gray-900">
                            Alerts & Notifications
                          </Dialog.Title>
                          <button
                            type="button"
                            className="rounded-md text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            onClick={() => setIsOpen(false)}
                          >
                            <span className="sr-only">Close panel</span>
                            <XMarkIcon className="h-6 w-6" aria-hidden="true" />
                          </button>
                        </div>
                      </div>
                      
                      {/* Alert List */}
                      <div className="flex-1 divide-y divide-gray-200 overflow-y-auto">
                        {alerts.map((alert) => (
                          <div
                            key={alert.id}
                            className={`p-4 hover:bg-gray-50 ${
                              !alert.isRead ? 'bg-blue-50' : ''
                            }`}
                          >
                            <div className="flex space-x-3">
                              <div className="flex-1 space-y-1">
                                <div className="flex items-center justify-between">
                                  <h3 className="text-sm font-medium text-gray-900">
                                    {alert.title}
                                  </h3>
                                  <span
                                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                      alert.type === 'error'
                                        ? 'bg-red-100 text-red-800'
                                        : alert.type === 'warning'
                                        ? 'bg-yellow-100 text-yellow-800'
                                        : 'bg-blue-100 text-blue-800'
                                    }`}
                                  >
                                    {alert.type}
                                  </span>
                                </div>
                                <p className="text-sm text-gray-500">
                                  {alert.message}
                                </p>
                                <p className="text-xs text-gray-400">
                                  {new Date(alert.timestamp).toLocaleString()}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Actions */}
                      <div className="border-t border-gray-200 p-4">
                        <div className="flex space-x-3">
                          <button
                            type="button"
                            className="flex-1 rounded-md bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
                          >
                            Mark All as Read
                          </button>
                          <button
                            type="button"
                            className="flex-1 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
                          >
                            Clear All
                          </button>
                        </div>
                      </div>
                    </div>
                  </Dialog.Panel>
                </Transition.Child>
              </div>
            </div>
          </div>
        </Dialog>
      </Transition.Root>
    </>
  );
} 